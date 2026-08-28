# Sprint 1 — Data and Model

**Goal:** turn the raw SemEval-2014 restaurant XML into a labelled dataset, fine-tune DistilBERT
on it, and save an artifact that reloads and predicts.

**Result:** working end-to-end. Validation macro-F1 0.5395, trained in 29.5 seconds on an RTX 4060.
No MLflow, no metrics suite, no API — those are later sprints on purpose.

---

## 1. The dataset, and one trap in it

SemEval-2014 Task 4 is the standard benchmark for aspect-based sentiment analysis. The restaurant
split is 3041 training sentences and 800 test sentences of real reviews, annotated by hand.

Each sentence looks like this:

```xml
<sentence id="3121">
  <text>But the staff was so horrible to us.</text>
  <aspectTerms>
    <aspectTerm term="staff" polarity="negative" from="8" to="13"/>
  </aspectTerms>
  <aspectCategories>
    <aspectCategory category="service" polarity="negative"/>
  </aspectCategories>
</sentence>
```

**Two different tasks live in that XML, and this project only does one of them.**

- `aspectTerms` are explicit spans — the literal word "staff", with character offsets. Predicting
  these is *aspect term extraction*: find which words name an aspect.
- `aspectCategories` are abstract labels from a fixed set. Predicting these is *aspect category
  detection and classification*.

This project does the second, so `aspectTerms` is parsed past and ignored. The distinction matters
because a category can be present with no term at all: *"it's overpriced"* is `price/negative`
without containing the word "price".

### The phase B trap

The official test data shipped in two files. `Restaurants_Test_Data_phaseB.xml` looks like the
labelled test set and is widely mirrored, but during the competition phase B handed competitors the
gold aspects and asked them to predict polarity — so its `polarity` attributes were **deliberately
stripped**. Parsing it yields 1025 annotations, every one with `polarity=None`.

The real labels live in `Restaurants_Test_Gold.xml`. Both files share identical sentence IDs, which
is how we verified the gold file is the same 800-sentence test set and not a different sample.

`data.py` fails loudly on a missing polarity rather than silently coercing it, and the error message
names phase B explicitly, because that is the mistake it is designed to catch.

### Two mismatches the loader reconciles

1. **Naming.** Upstream spells the categories `ambience` and `anecdotes/miscellaneous`; this project
   uses `ambience` and `misc`. Mapped explicitly in `CATEGORY_ALIASES` so the translation is
   auditable rather than buried in parsing logic.
2. **A fourth polarity.** Alongside positive/negative/neutral there is `conflict` — the sentence is
   both positive and negative about one aspect. About 5% of annotations. Handled below.

---

## 2. The central design decision: absence is a class

Most sentences mention only one or two of the five aspects. **2465 of the 3041 training sentences
mention exactly one.** So for any given sentence, most aspects were never discussed at all.

A plain three-class sentiment scheme has nowhere to record "this review never mentions price". The
model would be forced to assign positive, negative or neutral to an aspect the writer never raised.

So each aspect gets **four** classes:

```
0 = absent    1 = negative    2 = neutral    3 = positive
```

One forward pass answers both questions at once — *is this aspect discussed?* and *how does the
writer feel about it?* — instead of needing a separate detection stage.

**The alternative worth knowing about.** A well-known approach (Sun et al., 2019, "BERT-pair")
feeds the sentence once per aspect as a sentence pair: `[CLS] review [SEP] food [SEP]`. It scores
slightly better in the literature because the aspect name participates in attention. It also costs
**five forward passes per review instead of one**. For a system that serves predictions over HTTP,
that is a 5× latency and cost multiplier for a small accuracy gain. Being able to name the
trade-off is worth more than the accuracy.

### What happens to `conflict`

Three options, in increasing order of quality:

1. **Fold it into `absent`.** Actively harmful — it teaches the model that a clearly discussed
   aspect went unmentioned.
2. **Drop the whole sentence.** Wasteful. A sentence with `food=conflict, service=positive` still
   carries perfectly good information about service, and about the three aspects it doesn't mention.
3. **Mask just that one aspect.** What we do.

The label becomes `IGNORE_INDEX = -100`, which is the value PyTorch's `cross_entropy` skips by
default. That position contributes nothing to the loss and nothing to the gradient — the model is
neither taught nor penalised there. It is excluded from the metrics too, since scoring the model on
something it was never taught would be meaningless.

123 training sentences end up with every annotated aspect masked. They aren't wasted: their other
four aspects are still genuinely absent, and that is valid signal.

---

## 3. Architecture

```
        "Great food but slow service."
                    │
            DistilBERT encoder            66.4M parameters, shared
                    │
        last_hidden_state[:, 0]           the [CLS] vector, 768-d
                    │
            Linear(768 → 20)              15,380 parameters
                    │
         reshape (batch, 5, 4)
                    │
    ┌──────┬────────┼────────┬──────┐
   food  service ambience  price  misc     each a 4-way softmax
```

### Why one shared encoder

Five separate DistilBERTs would cost five times the memory and five times the inference latency,
and each would have to relearn English from a fifth of the supervision. Judging whether a sentence
criticises service and whether it praises food are largely the *same reading task* — the
representation transfers. This is the core intuition behind multi-task learning.

### Why the heads are one `Linear`, not five

`Linear(768, 20)` reshaped to `(batch, 5, 4)` is **mathematically identical** to five separate
`Linear(768, 4)` layers. Every output unit already owns its own independent row of weights, and no
aspect's logits can influence another's — there is no interaction term. One matrix multiply is
simply faster than five. If challenged on "that's not really five heads", this is the answer.

### Why `[CLS]`

BERT ships a "pooler" layer that produces a sentence vector. **DistilBERT does not** — verified
against the installed library, not assumed. So pooling is our choice. Position 0 is the `[CLS]`
token, whose representation attends over the whole sequence, making it the conventional sentence
summary. Mean-pooling over all tokens is the main alternative and sometimes works better; it wasn't
tested, and saying so honestly is better than implying it was.

### The 0.02% detail

The classification head is 15,380 parameters — **0.02% of the model**. Essentially all the learning
is DistilBERT's 66M weights adapting to restaurant reviews. This is what "fine-tuning" actually
means, and it is why the model works at all on 2,585 examples: it isn't learning language, it is
adjusting a model that already knows language.

---

## 4. The training loop

Written by hand rather than using HuggingFace's `Trainer`. `Trainer` gives checkpointing, logging
and distributed training for free, but hides the optimiser step, the scheduler and the evaluation
behind roughly a hundred constructor arguments. Forty explicit lines are worth more here.

Five choices worth being able to justify:

**Dynamic padding.** The `collate_fn` pads each batch only to its own longest sentence rather than
to a fixed 128 tokens. SemEval sentences are short, so fixed padding would spend most of the
compute on padding tokens.

**Weight decay excludes biases and LayerNorm.** Weight decay pulls weights toward zero as a
regulariser. Biases and LayerNorm scales need freedom to shift and rescale activations, so decaying
them hurts. Standard practice for transformer fine-tuning.

**Gradient clipping at norm 1.0.** Fine-tuning occasionally produces a gradient large enough to
destroy good weights in a single step. Clipping bounds the update size.

**Linear warmup then decay.** The learning rate ramps up over the first 10% of steps before
decaying. Starting at full learning rate on a pretrained model risks large early updates that wreck
the pretrained representation before the randomly-initialised head has learned anything useful.

**Every RNG seeded.** Python, NumPy and torch. Without this, two runs with identical
hyperparameters give different numbers and there is no way to distinguish an improvement from
noise — which matters immediately, since the next sprint compares MLflow runs.

### A free correctness check

Before training, the loss measured **1.3778**. An untrained 4-way classifier is maximally
uncertain, so its cross-entropy should equal `ln(4) = 1.3863`. Matching that confirms the reshape
and the `ignore_index` masking are wired correctly. Knowing to perform this check catches an entire
class of silent bugs.

---

## 5. Results, and what they actually mean

Four epochs, batch size 32, learning rate 2e-5:

| Epoch | train loss | val loss | macro-F1 |
|---|---|---|---|
| 1 | 0.8168 | 0.4991 | 0.3379 |
| 2 | 0.4263 | 0.3871 | 0.4644 |
| 3 | 0.3342 | 0.3473 | 0.5342 |
| 4 | 0.2937 | 0.3396 | 0.5395 |

Validation loss was still falling at epoch 4, so this is undertrained rather than overfit — an
honest baseline, not a tuned result. Tuning waits for MLflow, so runs can actually be compared.

### Why macro-F1 and not accuracy

`absent` is roughly 80% of all labels. A model that answers "absent" for everything, always, scores
about **76% accuracy** while being completely worthless. Macro-F1 averages the F1 of each class
equally, so the rare classes count as much as the common one, and that degenerate model scores
terribly. This is the clearest possible answer to "why not accuracy?", and it comes with real
numbers from this dataset.

### The headline number hides a real failure

Per-class F1 on the held-out test set, with support in brackets:

| Aspect | absent | negative | neutral | positive | macro |
|---|---|---|---|---|---|
| food | 0.915 (382) | 0.476 (69) | **0.000** (31) | 0.851 (302) | 0.561 |
| service | 0.979 (628) | 0.696 (63) | **0.000** (3) | 0.874 (101) | 0.637 |
| ambience | 0.970 (682) | 0.240 (21) | **0.000** (8) | 0.776 (76) | 0.496 |
| price | 0.979 (717) | 0.294 (28) | **0.000** (1) | 0.660 (51) | 0.483 |
| misc | 0.935 (566) | 0.044 (41) | 0.619 (51) | 0.687 (127) | 0.571 |

**The model never once correctly predicts `neutral`** for four of the five aspects. Not rarely —
zero times.

The cause is visible in the training data: `service` has 17 neutral training examples, `ambience`
19, `price` 9. There is nothing there to learn from. `misc` is the exception at 0.619, and it is
also the only aspect with substantial neutral training data (313 examples) — because
`anecdotes/miscellaneous` collects factual, non-evaluative statements like *"I live a block away"*.

A second failure: **`misc/negative` scores 0.044**, essentially broken, despite 41 test examples.
Worth investigating in the evaluation sprint.

So macro-F1 0.55 decomposes into: `absent` nearly solved (~0.95), `positive` respectable
(0.66–0.87), `negative` weak and inconsistent (0.04–0.70), `neutral` entirely absent. Excluding
neutral, macro-F1 would be roughly 0.72.

**Reporting the 0.55 and explaining this decomposition is far stronger than reporting a polished
number that conceals it.** Interviewers are considerably more interested in whether you know where
your model fails.

### Qualitative check

```
"The pasta was incredible but the waiter ignored us for twenty minutes."
  * food       positive   58.9%
    service    absent     37.2%     <- wrong, should be negative
```

It caught the food sentiment and missed the service complaint — but note the 37.2% confidence on
that miss. The model is genuinely torn, not confidently wrong. Compare with the aspects it
correctly rules out at 96–98%.

```
"I walked past it on my way to work."
  * misc       neutral    63.5%     <- correct
```

---

## 6. What the confidence number actually is

Each head produces a softmax over four classes; the confidence shown is the probability of the
chosen class. **This is not a calibrated probability of being correct.** Neural classifiers are
routinely overconfident — a set of predictions at "90% confidence" is usually right rather less
than 90% of the time.

This matters because the UI will display these as percentages to users. The honest framing is
"the model's relative preference among four options". Calibration (temperature scaling, reliability
diagrams) is a real technique and a legitimate thing to name as future work.

---

## 7. The three things most likely to be probed

**1. "Why four classes per aspect instead of three?"**
Because most sentences mention only one or two of the five aspects — 2465 of 3041 mention exactly
one — so the model needs a way to say "not discussed". Three-class sentiment would force a
sentiment onto aspects the writer never raised. Have the alternative ready too: a separate
detection stage followed by sentiment classification, which is two models and two failure modes
instead of one.

**2. "Your macro-F1 is 0.55. Is that good?"**
The trap is defending the number. The correct answer decomposes it: `absent` ~0.95, `positive`
~0.8, `negative` ~0.4, `neutral` 0.000 for four of five aspects because the training data contains
9–19 neutral examples for those aspects. The model isn't broken; the supervision isn't there. Then
name what you'd do: class weighting, oversampling, merging neutral into a coarser scheme, or
acquiring more data. **Knowing exactly where your model fails is the signal being tested.**

**3. "Walk me through what happens to a review at inference."**
Tokenised to at most 128 word-pieces → DistilBERT produces one 768-d vector per token → take
position 0, the `[CLS]` vector → one `Linear(768, 20)` → reshape to (5 aspects, 4 classes) →
softmax each row → argmax gives the label, the max probability gives the confidence. One forward
pass, all five aspects. Be ready for the follow-up: *why `[CLS]` and not mean pooling?* — because
`[CLS]` attends over the whole sequence and is conventional; mean pooling is a reasonable
alternative that wasn't tested here.

---

## 8. Deliberately not built yet

Named so they read as sequencing rather than omissions: MLflow tracking (sprint 2), confusion
matrices and a proper evaluation suite (sprint 3), explainability (sprint 3), the FastAPI service
(sprint 4), tests (sprint 5). Sprint 1 stops at "a saved model that reloads and predicts".

## 9. Files

| File | Purpose |
|---|---|
| `scripts/download_data.py` | Fetch + SHA-256 verify the dataset from pinned commits |
| `ml/src/data.py` | XML → per-aspect 4-way labels; category mapping; conflict masking; splits |
| `ml/src/model.py` | Shared DistilBERT + five 4-way heads; loss; batch prediction |
| `ml/src/train.py` | Fine-tuning loop, evaluation, artifact saving |
| `ml/src/predict.py` | Reload from disk and predict — the round trip the API depends on |

## 10. Known gaps carried forward

- `neutral` is unlearnable for four of five aspects on the current data.
- `misc/negative` at F1 0.044 needs investigation.
- Validation loss was still falling at epoch 4 — more epochs likely help.
- Mean pooling versus `[CLS]` was never compared.
- Confidences are uncalibrated.
