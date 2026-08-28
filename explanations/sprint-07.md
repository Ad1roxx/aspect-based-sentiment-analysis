# Sprint 7 — MAMS-ACSA as supplementary data

**Goal:** test the Sprint 6 conclusion that the multi-aspect failure is data-limited, by adding a
corpus built specifically for multi-aspect contrast.

---

## 1. Why this dataset

Sprint 6 gave the architecture hypothesis a fair test — three implementations, the mechanism verified
live — and the single–multi gap moved about one standard deviation. Combined with 77.9% of training
sentences mentioning exactly one aspect, that pointed at data rather than architecture.

MAMS-ACSA (Jiang et al., EMNLP-IJCNLP 2019, Apache-2.0) is the near-perfect probe, because it was
constructed by *deleting* every sentence with one aspect, or with multiple aspects sharing a
polarity. Verified rather than trusted:

```
sentences: 3949
aspects per sentence: {2: 3085, 3: 756, 4: 99, 5: 9}   mean 2.25
sentences with >=2 aspects           : 3949 (100.0%)
sentences with >=2 DISTINCT polarities: 3949 (100.0%)
```

Worth noting the published figures disagree with the files. One source says 8,879 sentences, the
survey paper says 5,297; the actual ACSA XML contains **3,949**. Counting the artefact you have beats
quoting the number someone wrote about it.

It is pinned in `scripts/download_data.py` to commit `cddcdb0` with SHA-256 verification, exactly
like the SemEval files.

---

## 2. Two problems the loader has to solve

### Overlap — measured, not assumed

MAMS derives from the same CSNY / Citysearch corpus as SemEval, so overlap was a real risk rather
than a theoretical one. Normalising both corpora (lowercase, strip punctuation, collapse whitespace)
and intersecting:

| | vs our train | vs our val | vs our test |
|---|---|---|---|
| MAMS train | 38 | 8 | **8** |
| MAMS val | 5 | 0 | 0 |
| MAMS test | 5 | 0 | **3** |

**67 sentences overlap ours; 11 are in our test split.** All 67 are dropped — including the train
overlaps, which would otherwise duplicate sentences under two different annotation schemes.

Training on those 11 would have raised the test score for the worst possible reason, and nothing in
the metrics would have looked wrong.

A detail worth keeping: a unit test asserting `normalise("It's good.") == normalise("Its good")`
failed, because apostrophes were being replaced with a *space*, turning `it's` into `it s` — which
never matches a corpus writing `its`. Apostrophes are now deleted while other punctuation still
becomes a space. The overlap count stayed at 67, which is reassuring: the figure is robust to the
normalisation choice rather than an artefact of it.

### Label semantics — the interesting one

MAMS `neutral` is not SemEval `neutral`:

| corpus | neutral rate (mentioned aspects) |
|---|---|
| SemEval-2014 | **13%** |
| MAMS | **43%** |

Per category: `menu` 79% neutral, `place` 60%, `food` 57%, `miscellaneous` 57%.

The cause is the construction rule. Every MAMS sentence must carry at least two *differing*
polarities, and `neutral` absorbed the slack — it functions closer to "mentioned, no strong opinion"
than to genuine neutral sentiment. Sampling makes it plain: every `miscellaneous` example inspected
was neutral, sitting alongside a real `staff` sentiment. One outright error turned up in the first
four examples read — *"I like the smaller portion size for dinner"* labelled **negative**.

Hence two modes, so the difference is measurable rather than arguable:

- **`filtered`** — MAMS positive/negative kept; MAMS neutral masked with `IGNORE_INDEX`, the same
  mechanism used for SemEval `conflict`. Masking is per *aspect*, not per sentence, so a sentence
  with one neutral and one negative still contributes the negative.
- **`full`** — everything kept. The naive merge.

**A correction worth recording.** The first design I proposed was to mask *every* mentioned aspect
and keep only the detection signal. That is incoherent: `IGNORE_INDEX` makes the loss skip the
position, so the model would learn nothing about those aspects — the opposite of teaching detection.
Detection is actually learned from the contrast with *unmentioned* aspects marked `absent`: the model
sees "here food is positive AND service is negative AND price is absent".

### The cost of filtering, which is not obvious

Masking 43% of annotations collapses most MAMS sentences back to single-aspect:

| mode | sentences with ≥2 mentioned aspects |
|---|---|
| `filtered` | 700 |
| `full` | 3,402 |

So `filtered` avoids the semantic mismatch but sacrifices most of the multi-aspect contrast that
motivated using MAMS at all — 2.6× more contrastive training data instead of 8.6×. That tension is
the whole reason both arms were run.

### Category mapping

MAMS has eight categories, this project has five. Two merges are judgement calls, flagged as such
rather than presented as obvious:

| MAMS | ours | confidence |
|---|---|---|
| food | food | safe |
| menu | food | **shaky** — 79% neutral, often really about a waiter knowing the menu |
| service, staff | service | safe |
| ambience | ambience | safe |
| place | ambience | **mixed** — "impressed by the room" yes, "just the place for you" is closer to misc |
| price | price | safe |
| miscellaneous | misc | safe, though 57% neutral |

Where two MAMS categories collapse onto one of ours with *different* polarities (staff vs service,
place vs ambience), the aspect is masked rather than letting whichever appeared last in the XML win —
otherwise the label would depend on file ordering. That happens 513 times.

---

## 3. Results — and my hypothesis was wrong

Three arms, three seeds each, identical config apart from the data:

| metric | `none` | `filtered` | `full` |
|---|---|---|---|
| val macro-F1 | 0.6047 ± 0.001 | 0.6122 ± 0.007 | **0.6247** ± 0.020 |
| multi-aspect recall | 0.7670 ± 0.016 | 0.7873 ± 0.016 | **0.8160 ± 0.000** |
| single−multi gap | 0.1253 ± 0.020 | 0.1170 ± 0.026 | **0.0710** ± 0.016 |

**The naive full merge won, and it is not close.** The gap falls from 0.1253 to 0.0710 — a **43%
reduction at 3.0 standard deviations.** For contrast, Sprint 6's architecture change managed 0.7
standard deviations, which is why it was reported as inconclusive.

Multi-aspect recall under `full` was **0.816 on all three seeds — standard deviation exactly zero.**
That kind of stability is itself evidence: the extra data is doing something systematic, not
something lucky.

Test macro-F1: **0.6287 (sprint 3) → 0.6315 (sprint 6) → 0.6389**, registered as version 6.

### I predicted the opposite

Section 2 argued MAMS's 43% neutral rate was "a different label doing a different job" and would
teach the model two incompatible meanings for one class. `filtered` was built to avoid exactly that.

It came second. The extra contrastive data was worth more than the label noise cost, and filtering
threw away most of the benefit — masking 43% of annotations collapsed 3,402 multi-aspect sentences
down to 700.

The distribution analysis was correct; the *inference* from it was wrong. That distinction matters:
observing that two datasets label differently does not tell you which way the trade lands. Only the
experiment does. Had I shipped `filtered` on the strength of the argument, the result would have been
a third of the available improvement and a confident explanation for why that was the right call.

### The sentence that started this

```
"The pasta was incredible but the waiter ignored us."
   food     positive  98.5%
   service  negative  95.3%     <- was: absent 46.3%
```

### An honest regression

```
"Cosy little place, though it is a bit overpriced for what you get."
   ambience  negative  64.5%    <- wrong; "cosy" is positive
   price     negative  83.5%    <- correct
```

The model now detects ambience here where it previously said nothing, but gets the polarity
backwards. The `place -> ambience` mapping is the likely culprit — it was flagged as shaky in §2,
60% of `place` annotations are neutral, and MAMS uses `place` for both "the room" and "the venue in
general". Better detection with worse polarity is a real cost of this merge, and it belongs in the
known-issues list rather than in a footnote.

---

## 4. What this settles

Sprint 6 concluded the multi-aspect failure was data-limited rather than architecture-limited, on the
strength of a change that did not work. That is weak evidence — "my fix failed, therefore the cause
is elsewhere" is an argument from absence.

This sprint is the positive test. Adding contrastive data moved the target metric 3.0 standard
deviations where the architecture change moved it 0.7. The two results together are a much stronger
claim than either alone:

> The multi-aspect failure is primarily a data-coverage problem. 77.9% of SemEval-2014 training
> sentences discuss exactly one aspect, and the model had almost no examples of "positive about X but
> negative about Y". Given those examples it learns the pattern; given a better pooling mechanism but
> the same data, it largely does not.

---

## 5. The three things an interviewer is most likely to probe

**(1) "You added a second dataset. How do you know it didn't leak into your test set?"**

The right answer starts with "I measured it, and it had": MAMS and SemEval both derive from the CSNY
corpus, 67 sentences overlapped, 11 were in the test split, and all 67 are dropped — train overlaps
too, since those would duplicate sentences under two annotation schemes. Then the detail that shows
care: the overlap check is normalisation-sensitive, a unit test caught apostrophes being turned into
spaces, and the count stayed at 67 after fixing it, so the figure is robust rather than an artefact.

**(2) "You built a filtered version and then didn't use it. Why?"**

Because it lost. This is the strongest honesty signal in the project: the analysis behind `filtered`
was correct — MAMS neutral really is 43% against SemEval's 13%, and really is being used as filler —
but the conclusion drawn from it was wrong, because filtering away 43% of annotations also destroyed
the multi-aspect contrast that made the dataset worth having. Both arms were run precisely so the
question could be settled by measurement rather than by whoever argued more confidently.

**(3) "Your macro-F1 only went from 0.6315 to 0.6389. Is that worth a whole dataset integration?"**

Do not defend macro-F1 — it is the wrong metric here and you should say so. It averages the failure
away: a lost aspect looks like a correct `absent` prediction alongside four other correct ones. The
metric built for this failure moved 43%, and multi-aspect recall was identical across three seeds.
Then close it honestly: detection improved, but the `place -> ambience` mapping introduced a polarity
regression, so the trade is not free.

---

## 6. Deliberately not built

- **Tuning the mix ratio.** MAMS is 3,882 sentences against SemEval's 2,585, so it now dominates
  training. Weighting or subsampling might do better; untested, and it would confound this result.
- **Fixing the `place` mapping.** Dropping `place`, or routing it to `misc`, is the obvious response
  to the regression above. It is a separate experiment with its own arm, not a patch to sneak in here.
- **SemEval-2016.** More data, but it needs a META-SHARE account and a mapping of the
  `ENTITY#ATTRIBUTE` scheme onto our five categories. Worth doing after this result is consolidated.
- **Retuning hyperparameters for the larger dataset.** Epochs and learning rate were tuned on 2,585
  sentences and are now training on 6,467. Legitimate, and it would confound the comparison.

---

## 7. Files

| File | Role |
|---|---|
| [`ml/src/mams.py`](../ml/src/mams.py) | Loader: category mapping, overlap removal, `filtered`/`full` modes |
| [`scripts/download_data.py`](../scripts/download_data.py) | MAMS pinned to `cddcdb0` with SHA-256 |
| [`ml/src/train.py`](../ml/src/train.py) | `--extra-data`; extra data joins TRAIN only |
| [`tests/test_mams.py`](../tests/test_mams.py) | Mapping, masking, merge conflicts, normalisation |

```
python scripts/download_data.py
python ml/src/mams.py
python ml/src/train.py --extra-data full --pooling attention --class-weights sqrt-inverse
```
