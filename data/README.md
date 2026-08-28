# Data

The SemEval-2014 Task 4 dataset is **not committed to this repository** — it is third-party
licensed. `data/raw/` is gitignored and populated by a script:

```bash
python scripts/download_data.py
```

Each file is pinned to an immutable upstream commit SHA and verified against a known SHA-256
digest, so the command is reproducible and fails loudly rather than training on unexpected data.

## Files fetched

| File | Sentences | Category annotations | Labelled? |
|---|---|---|---|
| `Restaurants_Train_v2.xml` | 3041 | 3713 | yes |
| `Restaurants_Test_Gold.xml` | 800 | 1025 | yes |

## Why the gold file, not `phaseB`

The official test release came in two parts, and picking the wrong one is an easy mistake:

- **`Restaurants_Test_Data_phaseB.xml`** — during the shared task, phase B gave competitors the
  gold aspect categories and asked them to predict the *polarity*. Its `polarity` attributes are
  therefore stripped. Verified directly: all 1025 annotations come back as `None`.
- **`Restaurants_Test_Gold.xml`** — the gold annotations released after the evaluation, with
  polarities intact.

Both files carry identical sentence IDs, which is how we confirmed the gold file is the same
800-sentence test set and not some other sample.

## Label distribution in the raw XML

Aspect categories, as spelled upstream:

| Category | Train | Test |
|---|---|---|
| `food` | 1232 | 418 |
| `anecdotes/miscellaneous` | 1132 | 234 |
| `service` | 597 | 172 |
| `ambience` | 431 | 118 |
| `price` | 321 | 83 |

Polarities:

| Polarity | Train | Test |
|---|---|---|
| `positive` | 2179 | 657 |
| `negative` | 839 | 222 |
| `neutral` | 500 | 94 |
| `conflict` | 195 | 52 |

Two things the loader has to reconcile with the project spec:

1. **Naming.** Upstream uses `ambience` and `anecdotes/miscellaneous`; this project uses
   `ambience` and `misc`. The loader maps them explicitly.
2. **`conflict` is a fourth polarity** (~5% of annotations) that the project's three-class scheme
   has no room for. Handled in `ml/src/data.py`.

Every sentence carries at least one category, but most carry only one — 2465 of 3041 training
sentences mention exactly one of the five aspects. That sparsity is what drives the model design.

## Licence

SemEval-2014 Task 4 data © the task organisers. See the
[task page](https://alt.qcri.org/semeval2014/task4/) for terms. Used here for research and
educational purposes.
