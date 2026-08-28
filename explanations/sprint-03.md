# Sprint 3 — Evaluation and explainability

**Goal:** turn Sprint 2's diagnosis into measured improvement, then add one honest, per-aspect
explainability method.

**Status:** done. Test macro-F1 **0.5497 → 0.6287** (+14.4% relative). Registered as
`absa-distilbert` version 3.

---

## 1. What Sprint 2 predicted, and whether it was right

Sprint 2 ended with two separate diagnoses. Sprint 3 tested both, and the distinction turned out to
matter enormously.

| # | Diagnosis | Predicted fix | Outcome |
|---|---|---|---|
| 1 | **Data sparsity.** `neutral` scores 0.000 wherever support is 1–10, and 0.541 on `misc` where support is 44. | Not fixable by tuning. | **Confirmed.** Class weighting did not move it. |
| 2 | **Calibration.** `misc/negative` at precision 1.000 / recall 0.031 — the model knows, but never commits. | Fixable by rebalancing the loss. | **Confirmed.** 0.061 → 0.567. |

Being able to say *"I predicted these were different problems with different fixes, then tested it
and one moved while the other didn't"* is worth far more than the score itself.

---

## 2. The class-weighting experiment

The training distribution across all (sentence, aspect) pairs:

| class | count | share |
|---|---|---|
| absent | 9,784 | 76.6% |
| positive | 1,844 | 14.4% |
| negative | 700 | 5.5% |
| neutral | 438 | 3.4% |

An imbalance of **22.3 : 1** between `absent` and `neutral`. With an unweighted loss, the gradient is
dominated by a class the model finds easy, and predicting a rare class is never worth the risk —
which is exactly the behaviour observed.

`F.cross_entropy` accepts a `weight` tensor that scales each class's contribution. Two schemes were
implemented rather than one, because "add class weights" is not a single decision:

- **`inverse`** — `w_c = N / (C · n_c)`, the textbook balanced weighting. Equalises each class's
  total contribution. Gives `neutral` a weight of 7.29.
- **`sqrt-inverse`** — the square root of the above, renormalised to mean 1. Corrects in the same
  direction with roughly a quarter of the force. Gives `neutral` 1.61.

Results, identical seed and config otherwise:

| scheme | val macro-F1 | val loss |
|---|---|---|
| `none` (baseline) | 0.5395 | 0.3396 |
| **`sqrt-inverse`** | **0.5980** | 0.3790 |
| `inverse` | 0.5233 | 0.6637 |

**Full inverse weighting scored below the baseline.** This is the result worth remembering: the
textbook fix, applied at full strength, made things worse. A weight of 7.29 pushes the model to
predict rare classes so eagerly that precision collapses, and macro-F1 punishes that as hard as it
punishes silence. The damped variant found the useful middle.

Anyone who says "just use class weights for imbalance" has not measured it.

### Where the gain actually came from

The headline number hides the real story. Per-class validation F1:

| aspect | `neutral` none → sqrt-inv | `negative` none → sqrt-inv |
|---|---|---|
| food | 0.000 → 0.000 | 0.581 → 0.707 |
| service | 0.000 → 0.000 | 0.603 → 0.716 |
| ambience | 0.000 → 0.000 | 0.421 → **0.357** ↓ |
| price | 0.000 → 0.000 | 0.457 → 0.679 |
| misc | 0.541 → 0.528 | 0.061 → **0.567** |

**Neutral did not improve at all.** Four aspects still score exactly 0.000. The entire gain came from
`negative` (and `positive`, which rose across the board).

That is diagnosis 1 holding up under test. With one to ten training examples of `price/neutral`,
there is nothing for a loss weight to amplify — you cannot reweight your way to data you do not
have. Fixing it needs more neutral data, or a coarser label space that stops pretending to
distinguish neutral from absent at that sample size.

Note also the honest regression: **`ambience/negative` got worse**, 0.421 → 0.357. Rebalancing is
not free, and reporting only the aspects that improved would be dishonest.

### Two design details worth defending

**Weights come from the training split only.** Computing them over train+val would leak the
validation distribution into a training decision. Small leak, but it makes the validation score
optimistic for no reason.

**Validation loss is deliberately left unweighted, even when training is weighted.** A loss computed
under different weights is a different quantity — comparing run A's weighted loss to run B's
unweighted loss is meaningless. Validation loss stays a fixed yardstick; macro-F1 is the selection
metric.

This shows up directly in the table above: the winning run has a **higher** val loss (0.3790 vs
0.3396) and a much better macro-F1. If you had selected on loss, you would have picked the worse
model. Understand why before an interview asks.

---

## 3. Registry hygiene: logging ≠ registering

Sprint 2 registered a new model version on every run. Run three experiments and the registry holds
three versions, two of which are experiments you would never deploy — and version numbers stop
meaning anything.

Now: every run still logs its own model, because that is what makes a run reproducible. But
promotion to a registry version is explicit, behind `--register`. The registry is a curated
shortlist of candidates.

The workflow that produced version 3:

```bash
# experiments — logged, not registered
python ml/src/train.py --class-weights sqrt-inverse --run-name weights-sqrt-inverse
python ml/src/train.py --class-weights inverse      --run-name weights-inverse

# compare, choose, then promote the winner
python ml/src/train.py --class-weights sqrt-inverse --eval-test --register
```

Note that `--eval-test` appears only on the final promotion run. The comparison runs were scored on
validation only — the test split was touched once, after the decision was already made.

---

## 4. `evaluate.py` — scoring what is actually on disk

Training already evaluates, so why a separate script?

Because training evaluates **the model it holds in memory**, and that is not the same object as the
artifact on disk. `evaluate.py` scores what was actually saved — the same files the API will load.
It is the only way to catch a broken save step, a tokenizer/weights mismatch, or an artifact
silently overwritten by a later run.

It also decouples evaluation from training: re-scoring a model six months later needs no GPU, no
training pipeline, and no 50 seconds of fine-tuning.

That it reproduced the training run's test macro-F1 **exactly** (0.6287) is the evidence that the
save/load path is faithful. A mismatch would have meant the API was about to serve something other
than the model that was measured.

### A refactor this forced

`per_class_metrics`, `classification_report_text` and `confusion_matrix_figure` lived in
`tracking.py`. But `evaluate.py` needs all three and needs **nothing** from MLflow, so importing
`tracking` would have dragged the whole experiment-logging apparatus in to draw a heatmap.

They moved to `evaluation.py`, which imports no MLflow at all. The seam is now:

- **`evaluation.py`** — *computes* things about predictions
- **`tracking.py`** — *records* them to MLflow

`tracking.py` shrank to 157 lines and does one job. This is the same lesson as Sprint 2's
`serving.py` split: the right module boundary showed up when a second consumer appeared.

### Final test-set results (version 3)

| aspect | macro-F1 |
|---|---|
| misc | 0.6850 |
| service | 0.6521 |
| food | 0.6477 |
| price | 0.6107 |
| ambience | 0.5482 |
| **overall** | **0.6287** |

`food/neutral` reached 0.111 on test (support 31) — still poor, but no longer exactly zero.
`service`, `ambience` and `price` remain at 0.000 with supports of 3, 8 and 1.

---

## 5. Explainability: why not attention

The obvious choice for a transformer is attention weights. It is the wrong choice **for this
architecture**, and the reason is worth being able to state precisely.

Our model pools a single `[CLS]` vector and feeds it to all five aspect heads. Last-layer attention
from `[CLS]` is therefore **one distribution shared by every aspect** — mathematically identical
whether you ask about food or about service. Rendering it on five aspect cards would produce five
identical highlights. That is worse than useless: it *looks* like a per-aspect explanation while
being nothing of the sort.

There is also a general objection — Jain & Wallace (2019), *"Attention is not Explanation"* — showing
attention weights can be substantially altered without changing the prediction, so they do not
reliably indicate what a model depended on. But the architectural argument is the decisive one here,
and it is the stronger thing to say, because it is specific to what we built.

### What was built instead

**Gradient × input attribution.** For a chosen aspect and its predicted class:

```
attribution(token) = Σ_d ( ∂logit / ∂embedding[token, d] ) · embedding[token, d]
```

The gradient measures how sharply that logit responds to that position; dotting with the actual
embedding converts sensitivity into a contribution from the value actually present.

Crucially, the gradient is taken from **one aspect's logit**, so a different aspect produces a
different gradient and a genuinely different explanation. Verified on *"The sushi was fresh and
delicious but our waiter was incredibly rude"* — the food and service rankings differ substantially.

Cost is one forward and one backward pass, which is what makes it viable inside an API request.

Implementation notes worth knowing:

- A **forward hook** captures the embedding output rather than changing `forward()` to accept
  `inputs_embeds`. Explainability observes the model; it does not get to reshape its interface.
- `retain_grad()` is required — the embedding output is a non-leaf tensor, so PyTorch discards its
  `.grad` after backward unless told otherwise.
- **Subwords are merged.** DistilBERT splits "overpriced" into `over`, `##pric`, `##ed`. Showing
  fragments is unreadable and splits one word's contribution three ways so each looks minor.
  Attributions are additive, so summing fragments is the correct aggregation.
- `[CLS]` and `[SEP]` are dropped — `[CLS]` accrues large attribution simply for being the pooled
  position, which says nothing about the text.

### The sign is discarded, and that decision was measured

The raw attribution is signed, and the first implementation reported the sign. It was wrong.

On *"…a bit overpriced for what you get"*, the word **overpriced** received the largest magnitude —
correctly — but with a **negative** sign on a `price → negative` prediction, implying it argued
against the class it obviously drove.

The cause is structural. Gradient × input approximates `f(x) − f(0)`: a Taylor expansion around an
**all-zero embedding**. The zero vector is not a token, so "contribution relative to the zero
embedding" has no linguistic meaning and its sign is an artefact of an arbitrary baseline. (This is
precisely the problem Integrated Gradients solves, by integrating along a path from a *chosen*
baseline such as `[MASK]` or `[PAD]`.)

What survives is magnitude. To check that magnitude was trustworthy rather than assume it, the top
word was occluded and the model re-run:

| input | prediction |
|---|---|
| "…a bit **overpriced** for what you get" | price **negative** 69.8% |
| "…a bit **[MASK]** for what you get" | price **absent** 75.6% |
| "The pasta was **incredible** but…" | food **positive** 73.0% |
| "The pasta was **[MASK]** but…" | food **negative** 29.4% |

Both flip. The ranking identifies words the model genuinely depends on.

Occlusion is the more trustworthy method — it is causal rather than a linear approximation — but it
costs one forward pass **per word**. Using the cheap method in production and validating it against
the expensive one offline is the trade, and it is a defensible one.

### Known artefacts, stated rather than hidden

**Punctuation dominates.** The final `.` ranked #1 for both aspects in the two-aspect test. Tokens
adjacent to `[SEP]` commonly accumulate attribution. It is *not* filtered from the output, because
filtering would conceal real model behaviour — the UI can choose to hide it, but the method should
not lie about it.

**The model was wrong in that example.** For *"our waiter was incredibly rude"* it predicted
`service → positive` at 40.4%. The explanation faithfully explains a wrong prediction, which is
exactly what an explanation method should do. The low confidence is at least appropriate. Do not
present explainability as evidence the model is correct.

---

## 6. The three things an interviewer is most likely to probe

**① "You added class weights and it improved. How do you know it was the weights?"**

The answer is the experiment design, not the number: same seed, same config, one variable changed,
three runs recorded in MLflow with per-class metrics. Then the sharper point — **full inverse
weighting scored *below* baseline (0.5233 vs 0.5395)**, so the direction was right but the magnitude
mattered more than the idea. Then the honest decomposition: the gain came entirely from `negative`
and `positive`; `neutral` did not move, because it is a data problem, not a loss problem. And name
the regression — `ambience/negative` fell 0.421 → 0.357 — before they find it.

**② "Why gradient × input rather than attention?"**

Lead with the architectural argument, because it is specific to this model and cannot be recited
from a blog: one `[CLS]` vector feeds five heads, so `[CLS]` attention is *shared across aspects*
and would render five identical highlights. Then the empirical follow-up: the sign was discarded
because it was measured to be untrustworthy, and magnitude was validated by occlusion flipping both
demo predictions. Have Integrated Gradients ready as the "what would you do with more budget"
answer, and be able to say why — a real baseline instead of the zero vector.

**③ "Your macro-F1 is 0.63. What's still broken, and what would you do next?"**

The trap is claiming it is fine. `neutral` is still 0.000 on service, ambience and price, with test
supports of 3, 8 and 1. That is not a model failure so much as a labelling scheme that pretends to
make a distinction the data cannot support. Concrete options, in order of honesty: merge
`neutral` into `absent` for the sparse aspects and say so explicitly; collect or augment neutral
examples; or keep four classes and report per-class results rather than hiding behind a macro
average. Volunteering that the current label space may be wrong is a stronger answer than promising
a better learning rate.

---

## 7. Deliberately not built

- **Integrated Gradients / SHAP / LIME.** One explainability method was the scope. IG is the natural
  upgrade and is named as future work with a reason, not just a buzzword.
- **Threshold tuning per class.** A real lever — you can trade precision for recall directly — but it
  interacts with class weighting and would confound the experiment just run.
- **Focal loss.** Another imbalance approach. Nothing here justifies a third scheme when the second
  already beat the first.
- **Per-aspect class weights.** Currently one weight vector applies across all five aspects, because
  the loss is one flattened cross-entropy call. Per-aspect weights would need five separate calls.
  Noted as a limitation in `data.class_counts`, not silently ignored.
- **Cross-validation.** Would give error bars on 0.6287 instead of a point estimate. Correct, and
  5× the compute; a single seeded split is the deliberate simplification.

---

## 8. Files

| File | Role |
|---|---|
| [`ml/src/evaluation.py`](../ml/src/evaluation.py) | Computes per-class metrics, report tables, confusion grids. No MLflow. |
| [`ml/src/evaluate.py`](../ml/src/evaluate.py) | CLI: scores the saved artifact on any split. |
| [`ml/src/explain.py`](../ml/src/explain.py) | Per-aspect gradient × input attribution. |
| [`ml/src/train.py`](../ml/src/train.py) | `--class-weights`, `--register`, `--run-name` added. |
| [`ml/src/tracking.py`](../ml/src/tracking.py) | Now purely a recorder; registration is opt-in. |

```bash
python ml/src/evaluate.py --split test --save-figure
python ml/src/explain.py --text "Great food, awful service." --aspect service
mlflow ui --backend-store-uri sqlite:///ml/mlflow.db
```

---

## 9. Carried into Sprint 4 (API)

1. **The two load paths must be resolved.** `predict.load_model()` reads disk;
   `mlflow.pyfunc.load_model("models:/absa-distilbert/3")` reads the registry. The API has to pick
   one and the choice needs a stated reason.
2. **Explainability needs a response shape.** `explain()` returns `(word, importance)` pairs; the API
   contract and the React highlighting depend on that shape.
3. **Explainability cost.** One backward pass *per aspect* — five per request if all are explained.
   Explaining only detected aspects, or only on request, is a decision the endpoint design must make.
4. **Still no tests.** The occlusion check, the save/load fidelity check and the registry round-trip
   were all verified by hand. All three should be pytest cases in Sprint 5.
5. **`neutral` remains unsolved** and should be described honestly in the README rather than averaged
   away.
