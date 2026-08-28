# Sprint 6 — Per-aspect attention pooling

**Goal:** replace the shared `[CLS]` pooling so each aspect head attends to its own evidence, and
rename `ambiance` → `ambience` to match the dataset.

**Status:** done. Test macro-F1 **0.6287 → 0.6315**, registered as version 5. But the honest headline
is different: **the architecture change did not fix the multi-aspect problem it was built to fix**,
and finding that out required fixing two bugs that every accuracy metric was blind to.

---

## 1. Where this sprint came from

A prediction that looked obviously wrong:

```
"The pasta was incredible but the waiter ignored us."
  service -> absent (0.4629)
```

Investigating it produced a much sharper diagnosis than the one on file:

| sentence | service |
|---|---|
| "The waiter ignored us." | negative 0.73 ✓ |
| "Our waiter was rude." | negative 0.79 ✓ |
| "The pasta was incredible **but** the waiter ignored us." | absent 0.46 ✗ |

The model reads service *perfectly in isolation*. It fails when a competing aspect is present.
Measured across the test split — of aspects genuinely discussed, how often the model avoids calling
them absent:

- **single-aspect sentences: 91.8%**
- **multi-aspect sentences: 78.8%**

That 13-point gap is what this sprint attacked. The hypothesis: one shared `[CLS]` vector feeds all
five heads, so a strong signal for one aspect crowds out a weaker competing one. The fix: give each
aspect its own learned query and let it attend over the tokens itself.

A new metric was added first, because **macro-F1 averages this failure away** — a lost aspect looks
like a correct `absent` prediction alongside four other correct `absent`s. `mention_recall_single`,
`mention_recall_multi` and their gap are now logged per run.

---

## 2. Two bugs that no accuracy metric could see

This is the part worth understanding, and it nearly went unnoticed.

### Bug 1 — the mechanism was mean pooling in disguise

The first implementation used scaled dot-product attention, copying the standard formula:

```python
scores = einsum("btd,ad->bat", hidden_states, aspect_queries) / sqrt(hidden_size)
```

Trained. Compared. Result: within noise of the `[CLS]` baseline. The obvious reading is "the idea
didn't work". The actual reading, found only by printing the weights:

```
attention: min=0.0764 max=0.0775   uniform would be 0.0769
max/min ratio = 1.014
```

**Every weight was uniform.** The attention was mean pooling wearing a costume. It had never been
tested at all.

The cause: `1/sqrt(d)` exists to stop softmax *saturating* when queries and keys both have
unit-variance entries. Here the "keys" are raw encoder hidden states (norm ≈ 16.6) and the queries
start at `std=0.02` (norm ≈ 0.55), so raw scores already spanned only ~1.07. Dividing by
`sqrt(768) = 27.7` compressed that to 0.038, and a softmax over a 0.038 range is flat.

Worse, it shrank the gradient reaching the queries by the same factor.

### Bug 2 — the queries could not learn their way out

With the scaling removed, attention could differentiate at initialisation (ratio 2.60). Retrained.
Still barely moved:

```
query norms after training: [0.55, 0.55, 0.56, 0.54, 0.54]
query norms at initialisation: 0.55
```

**The queries had not learned anything.** Learning rate `2e-5` is tuned for *nudging a pretrained
encoder*; a randomly-initialised parameter that has to travel a long way in 324 steps needs far
more. The fix is standard practice and should have been there from the start — a separate parameter
group:

```python
{"params": fresh_params, "lr": config.query_learning_rate}   # 1e-3, vs 2e-5
```

After which:

| | broken | scaling fixed | + own learning rate |
|---|---|---|---|
| query norm | 0.55 (= init) | 0.55 (= init) | **0.66 – 0.91** |
| attention max/min | 1.014 | 1.53 | **183** |

**The generalisable lesson:** accuracy metrics cannot tell you that a component is inert. Both bugs
produced a model that trained fine, scored fine, and had a dead mechanism inside it. Only inspecting
the intermediate values found them. If you add a mechanism, assert on *the mechanism*, not on the
metric downstream of it.

That is exactly what the new regression tests do — `tests/test_model.py::TestAttentionPooling`
asserts the weights are non-uniform, differ between aspects, and give padding exactly zero mass.
They run on synthetic tensors with no encoder, because `attention_pool` was extracted to a
module-level function for that purpose.

---

## 3. The result, honestly

Three seeds each, identical config, only the pooling differing:

| metric | `cls` | `attention` | delta |
|---|---|---|---|
| val macro-F1 | 0.6024 ± 0.0042 | 0.6047 ± 0.0011 | +0.0023 |
| multi-aspect recall | 0.7517 ± 0.0161 | **0.7670** ± 0.0157 | **+0.0153** |
| single−multi gap | 0.1380 ± 0.0130 | **0.1253** ± 0.0198 | **−0.0127** |

Test macro-F1 went 0.6287 → 0.6315.

**Read this carefully, because the temptation is to over-claim.** The gap improved by 0.0127 with a
pooled standard deviation of 0.0164. The effect is *smaller than the noise*. Three seeds cannot
establish it. The direction is right, the magnitude is not convincing, and the honest summary is:

> Per-aspect attention gives a small, directionally-correct improvement in multi-aspect recall that
> is not statistically separable from seed variance at n=3. It did not close the gap.

Two things did clearly improve, and both are qualitative:

- The motivating sentence is now correct. "The sushi was fresh and delicious but our waiter was
  incredibly rude" gives `service: negative 59.1%`, where the `[CLS]` model said `positive 45.7%`.
- Variance across seeds fell sharply on macro-F1 (±0.0011 vs ±0.0042).

### Why keep it, then

Not for accuracy — the evidence does not support that claim. It is kept because the attention
weights are a genuinely per-aspect explanation computed **free in the forward pass**:

```
"The sushi was fresh and delicious but our waiter was incredibly rude."
  food    -> positive   the=0.19  fresh=0.16  delicious=0.15
  service -> negative   was=0.21  rude=0.17   our=0.14
```

This retires the Sprint 3 objection. Attention was rejected then because one `[CLS]` vector fed all
five heads, so `[CLS]` attention was mathematically identical for every aspect. That is no longer
true, and it cost 3,840 parameters (5 x 768, or 0.006% of the model).

A head-to-head against gradient x input, on 40 test sentences, occluding each method's top-ranked
word and measuring the confidence drop:

| method | mean drop | median | head-to-head wins |
|---|---|---|---|
| attention | +0.215 | +0.063 | 16/40 |
| gradient x input | +0.175 | +0.063 | 24/40 |

**Comparable, not better.** Attention has a higher mean pulled up by a few large wins, an identical
median, and loses most direct comparisons. So gradient x input stays the shipped method — it is
already tested and wired into the API — and attention is noted as a cheaper alternative (no backward
pass) worth a proper evaluation later. Switching on the strength of a higher mean and a worse
win-rate would be exactly the kind of selective reading this project tries to avoid.

---

## 4. So: is more data needed for multi-aspect sentences?

**Yes, and this sprint is the evidence.**

The architecture hypothesis has now been given a fair test — three implementations deep, with the
mechanism verified to be working (attention ratio 183, queries demonstrably learning) — and the gap
moved by roughly one standard deviation. If the shared-`[CLS]` bottleneck were the dominant cause,
fixing it should have produced far more than that.

What the training data actually contains:

| aspects mentioned | sentences | share |
|---|---|---|
| 0 | 123 | 4.8% |
| **1** | **2,014** | **77.9%** |
| 2 | 381 | 14.7% |
| 3 | 62 | 2.4% |
| 4 | 5 | 0.2% |

Only **448 multi-aspect training sentences**, and the contrastive case that fails — two aspects with
*opposing* sentiment — is a subset of those. The model has seen very few examples of "positive about
X but negative about Y", which is precisely the pattern it gets wrong.

We already use 100% of SemEval-2014 restaurants. Options, in order of cost:

1. **Synthetic contrastive augmentation.** Join two single-aspect training sentences with "but" and
   carry both labels across. 2,014 single-aspect sentences make an enormous pool of pairs. Cheap, and
   directly targets the failing pattern. Risk: templated sentences are less natural than real ones,
   and the model may learn the template rather than the contrast — so it needs a held-out *real*
   multi-aspect evaluation, which we have.
2. **SemEval-2016 Task 5** adds restaurant data in a related annotation scheme; usable with mapping
   work.
3. **Manual annotation** of multi-aspect reviews. Most faithful, slowest.

Option 1 is the obvious next experiment, and the instrument to judge it already exists:
`mention_recall_multi` is tracked per run, so "did it help?" is a query rather than an argument.

---

## 5. The rename

SemEval-2014 spells the category `ambience`. An alias renamed it to `ambiance` to match the original
project brief, which meant the API exposed a field name differing from the source data for no
technical reason — a thing you then have to explain forever.

The alias was removed rather than inverted. Four of five categories are now used exactly as SemEval
spells them, and only `anecdotes/miscellaneous` becomes `misc`, because that one is genuinely
unusable as an API field name and a UI label.

Model weights were unaffected: the aspect is still index 2.

---

## 6. The three things an interviewer is most likely to probe

**(1) "You changed the architecture and the metric barely moved. Was it worth it?"**

Do not defend the metric. The sprint produced three things worth more than the 0.0028 test-set gain:
two bugs that made a component inert while every metric looked healthy, a regression suite that
asserts on the mechanism instead of the metric, and — most usefully — evidence that the multi-aspect
problem is *data-limited rather than architecture-limited*, which determines the next experiment. A
negative result you can act on beats a positive one you cannot explain.

**(2) "Walk me through the attention bug."**

The strongest story in the project. `1/sqrt(d)` prevents softmax saturation when queries and keys
have unit-variance entries; here the keys were raw hidden states and queries started at std=0.02, so
scores already spanned about 1 nat, and dividing by 27.7 flattened the softmax to uniform — max/min
ratio 1.014, mean pooling — while shrinking the gradient by the same factor so it could never
recover. Then the second half: even unscaled, the queries would not move at lr=2e-5, because that
rate is for nudging a pretrained encoder, not training a fresh parameter. Both were invisible in
accuracy and immediately visible in the weights.

**(3) "How do you know it's a data problem and not your model?"**

Because the alternative was tested rather than assumed. Three implementations, the mechanism verified
live, three seeds, and a metric built specifically to measure the failure (`mention_recall_multi`,
since macro-F1 averages it away). The gap moved about one standard deviation. Combined with 77.9% of
training sentences being single-aspect and only 448 multi-aspect examples in total, the conclusion
follows from measurement. Then name the next experiment — synthetic contrastive augmentation — and
how you would judge it.

---

## 7. Files

| File | Change |
|---|---|
| [`ml/src/model.py`](../ml/src/model.py) | `attention_pool`, switchable pooling, `return_attention` |
| [`ml/src/train.py`](../ml/src/train.py) | `--pooling`, separate LR group for fresh params, mention-recall metrics |
| [`ml/src/predict.py`](../ml/src/predict.py) | rebuilds with the pooling recorded in metadata |
| [`ml/src/serving.py`](../ml/src/serving.py) | same, for the registry path |
| [`ml/src/data.py`](../ml/src/data.py) | `ambience`; alias removed |
| [`tests/test_model.py`](../tests/test_model.py) | `TestAttentionPooling` — the regression tests |
| [`tests/test_integration.py`](../tests/test_integration.py) | occlusion test fixed |

```
python ml/src/train.py --pooling attention --class-weights sqrt-inverse
python ml/src/train.py --pooling cls       --class-weights sqrt-inverse
```

---

## 8. Carried into Sprint 7 (frontend)

1. **Confidence is a usable UI signal.** A *lost* aspect sits near 0.46-0.52; a genuinely absent one
   sits at 0.93-0.96. The UI can legitimately treat a low-confidence `absent` as "possibly discussed"
   rather than silently hiding it.
2. **Attention weights are available** via `forward(..., return_attention=True)` at no extra cost, if
   the highlighting UI wants a cheaper source than gradient x input.
3. **`pooling` is load-bearing metadata.** An artifact cannot be rebuilt without it, and guessing
   wrong loads a state dict into the wrong architecture rather than erroring.
4. **The multi-aspect gap is unresolved and now well-characterised** — the honest thing for the
   README, not something to average away.
