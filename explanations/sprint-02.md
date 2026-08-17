# Sprint 2 — MLflow experiment tracking

**Goal:** make every training run reproducible and comparable — params, per-class metrics,
confusion matrices and a version-numbered model artifact, recorded rather than printed.

**Status:** done. Two runs recorded, `absa-distilbert` versions 1 and 2 in the Model Registry,
both loadable by URI.

---

## 1. The problem this actually solves

Sprint 1 printed metrics to the terminal. That is entirely adequate for exactly one run and
useless for two.

The moment tuning starts — and it is about to, because four of five aspects score 0.000 on
`neutral` — questions appear that scrollback cannot answer:

- Which run had learning rate 3e-5, and did it help or hurt?
- The run that scored 0.58 — was that with class weights, or was that the seed?
- We changed three things and it improved. Which one mattered?
- Six weeks from now, which code produced the artifact currently being served?

Every one of those is a *bookkeeping* problem, not a modelling problem, and bookkeeping problems
are the ones that quietly destroy ML projects. A model you cannot reproduce is a model you cannot
improve, because you can never tell an improvement from noise.

MLflow is the ledger. Three things go into it per run:

| Kind | What | Why |
|---|---|---|
| **params** | the inputs — `Config` plus encoder, `max_length`, step count, parameter count | enough to re-run it |
| **metrics** | loss and F1 per epoch, then per-aspect and per-class F1 with support | the numbers you compare |
| **tags** | device, GPU model, torch/python versions, split sizes | the context you need to *diagnose* it |
| **artifacts** | confusion-matrix grid, P/R/F1 table, the model itself | the evidence and the output |

The params/tags distinction is worth internalising: **params are what you chose, tags are what was
true.** You did not choose to have an RTX 4060; it was a fact about the run. Both are filterable in
the MLflow UI, so "show me every run with `lr=3e-5` on cuda" is a query rather than a memory test.

---

## 2. Why the backend store is sqlite, not `./mlruns`

Nearly every MLflow tutorial written before 2025 uses the default filesystem store — a `mlruns/`
directory of YAML files. We use `sqlite:///ml/mlflow.db` instead, for two reasons, and I verified
both rather than assuming them:

1. **The file store now refuses to run.** On MLflow 3.x it raises outright:

   > The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive
   > further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db')…

   You can opt out with `MLFLOW_ALLOW_FILE_STORE=true`, but opting out of the supported path to keep
   a deprecated one is a bad trade.

2. **The file store never supported the Model Registry.** The registry is the component that assigns
   version numbers. Without it there is no such thing as `models:/absa-distilbert/2`, and "versioned
   artifact" degrades to "a folder I promise I did not overwrite".

The artifact location is also pinned explicitly, to `ml/mlartifacts`. Left alone, MLflow drops an
artifact directory wherever the process happened to start, so running training from the repo root
and from `ml/` scatters artifacts into two places. `configure()` creates the experiment with an
explicit `artifact_location` on first use to prevent that.

A note on scale, since an interviewer may push on it: sqlite is a *local development* backend. A
team would run a tracking server backed by Postgres with artifacts in S3. The code change is one
line — the tracking URI — which is exactly why the abstraction is worth having.

---

## 3. The bug worth understanding: `ModuleNotFoundError: No module named 'tracking'`

This is the most instructive thing in the sprint, so it is documented rather than quietly fixed.

MLflow's Model Registry will not accept a plain directory of files. I checked:

```
mlflow.log_artifacts('fake_model', artifact_path='model')
mlflow.register_model(f'runs:/{run_id}/model', 'probe-absa')
→ MlflowException: Unable to find a logged_model with artifact_path model under run …
```

So the model must be logged as a real MLflow model, which for a custom architecture means a
`mlflow.pyfunc.PythonModel` wrapper. Fine. I wrote `ABSAModel` inside `tracking.py`, training ran,
version 1 registered successfully — and then loading it failed:

```
mlflow.pyfunc.load_model('models:/absa-distilbert/1')
→ ModuleNotFoundError: No module named 'tracking'
```

**Why:** MLflow serialises a `PythonModel` with cloudpickle, which stores the class **by reference**
— module name plus qualified name — not the class body. So the artifact contains, in effect, "the
class `ABSAModel` from the module `tracking`". At load time MLflow rebuilds it in a fresh
environment where the only importable code is whatever `code_paths` copied in. `code_paths` listed
`model.py` and `data.py`. It did not list `tracking.py`. The reference dangled.

Two ways out:

- **Add `tracking.py` to `code_paths`.** Works, and is wrong: it drags matplotlib, sklearn and the
  entire experiment-logging apparatus into every serving container to support a class that uses none
  of them.
- **Move the wrapper to its own module.** `serving.py` now holds `ABSAModel` and imports only
  `mlflow`, `torch`, and (lazily) `transformers` + `model`.

The second is right because it follows a real seam: **`serving.py` is what ships inside the
artifact; `tracking.py` is build-time only and stays behind.** The bug forced the correct
architecture, which is often how it goes.

The generalisable lesson: **registering an artifact is not the same as verifying it loads.** The
registry happily accepted a broken model. Only a round-trip test caught it. Any "we log models to
MLflow" claim that has never been load-tested is roughly 50% likely to be false.

`code_paths` now lists three files, each needed for a different reason:

| File | Why it must ship |
|---|---|
| `serving.py` | defines the class the pickle refers to |
| `model.py` | defines the architecture `load_context` rebuilds |
| `data.py` | supplies `ASPECTS` / `LABEL_NAMES` that `model.py` imports |

Omit any one and the failure appears only at load time — never at log time.

> Files are listed individually rather than passing the `src/` directory. A directory would nest
> imports one level deeper (`from src.model import …`) and break `model.py`'s own `from data import`.

---

## 4. The confusion matrices, and one deliberate visual decision

Five aspects, each a 4×4 matrix, drawn as a 2×3 grid with the sixth cell hidden.

The decision worth defending: **cells are annotated with raw counts but coloured by row-normalised
proportion.**

Colouring by raw count would produce a useless picture. `absent` outnumbers every sentiment class by
roughly an order of magnitude, so one cell in the top-left would be black and the other fifteen
would be indistinguishable white. Normalising each row by its true-class total asks a better
question: *"of the examples that really were X, where did they go?"* — which is recall per class,
and recall is precisely what collapses here.

Row totals are clamped with `np.maximum(row_totals, 1)` so a class absent from a split renders as an
empty row rather than `NaN`.

**Known weakness of this choice, stated plainly:** with tiny support, normalisation lies. The
`price` / `neutral` row has exactly **one** validation example. It was predicted `negative`, so that
cell renders 1/1 = 100% — the darkest possible square, visually implying a confident correct
behaviour when it is one example going the wrong way. Read the counts, not just the colour. Overlaying
support counts on the row labels would fix this and is a reasonable Sprint 3 refinement.

---

## 5. Test-set hygiene — why `--eval-test` is opt-in

`Config.eval_test` defaults to `False`.

The test split exists to estimate performance on data that never influenced the model. Every time
you look at it and then change something, you leak a little information from it into your decisions.
Do that on every run through a tuning loop and the test set silently becomes a second validation set
— the number keeps being reported, but it has stopped meaning what it claims to mean. This is a
slow, invisible failure mode, and it is *extremely* common in student projects.

So: validation is scored every epoch, automatically. Test is scored only when explicitly asked for.
It was enabled for the two runs here because they establish the baseline rather than tune against it
— the model, seed and config are identical to Sprint 1's.

That reproduction is itself a result worth noting: **epoch-by-epoch numbers were identical to Sprint
1** (0.3379 → 0.4644 → 0.5342 → 0.5395). Refactoring `evaluate()` to return raw predictions, and
wrapping everything in an MLflow run, changed no training behaviour. That is what `set_seed` is for,
and it is the first time the seeding has actually been tested.

---

## 6. What the tracking immediately revealed

This is the payoff. Sprint 1 knew "macro-F1 is 0.5395 and neutral is bad". The per-class report says
something much more specific:

```
food      neutral   prec 0.000  recall 0.000  f1 0.000   support 10
service   neutral   prec 0.000  recall 0.000  f1 0.000   support  3
ambiance  neutral   prec 0.000  recall 0.000  f1 0.000   support  4
price     neutral   prec 0.000  recall 0.000  f1 0.000   support  1
misc      neutral   prec 0.561  recall 0.523  f1 0.541   support 44
```

**Finding 1 — the neutral collapse is mostly a data problem.** Neutral works exactly where neutral
data exists. With 44 validation examples `misc` reaches 0.541; with 1–10 examples the other four
reach zero. The confusion matrices confirm it structurally: the `neutral` *column* is entirely empty
for food, service, ambiance and price. The model never predicts neutral for those aspects — not
rarely, never. You cannot fix that with a learning rate.

**Finding 2 — `misc/negative` is a threshold problem, not a knowledge problem.**

```
misc  negative  prec 1.000  recall 0.031  f1 0.061  support 32
```

Precision 1.000 with recall 0.031: of 32 truly-negative misc examples it predicted negative once,
and was right. The model *can* identify these; it is simply so biased toward the majority class that
it almost never commits. The confusion matrix shows those 32 going 12→absent, 16→**positive**, 3→neutral.
Being wrong toward *positive* is worse than being wrong toward *absent* — the product surfaces a
confident wrong sentiment rather than staying silent.

**Finding 3 — recall sits below precision on essentially every sentiment class.** That is the
signature of majority-class bias from an ~80% `absent` prior. It points at concrete Sprint 3 levers:
class weighting in the loss, threshold tuning, or restructuring absence detection as a separate head.

Note the end-to-end symptom too. On *"The pasta was incredible but the waiter ignored us for twenty
minutes"* the model returns `food/positive` and misses `service/negative` entirely — the exact
failure the numbers predict.

---

## 7. The three things an interviewer is most likely to probe

**① "You log a model to MLflow — walk me through what's actually in that artifact, and how the
serving code gets it back."**

This is where vague answers die. The real one: a pyfunc-wrapped model containing the state dict, the
tokenizer, `metadata.json`, an inferred input signature (`[string (required)]`), a conda/pip
environment spec, and three source files copied via `code_paths`. It loads with
`mlflow.pyfunc.load_model("models:/absa-distilbert/2")`. Be ready for the follow-up — *"why the
wrapper?"* — because the answer is concrete: `register_model` rejects a bare directory, and the
registry is where version numbers come from. And be ready to volunteer the `ModuleNotFoundError`
story from §3; a bug you diagnosed and fixed structurally is stronger evidence of understanding than
a component that happened to work.

Related, and worth pre-empting: **there are currently two load paths.** `predict.load_model()` reads
`ml/models/absa-distilbert/` directly, and the MLflow registry serves versioned artifacts. That is a
deliberate, temporary state — the API sprint has to pick one. Know which you would pick and why
(registry, because it pins a version; direct load, because it has no MLflow dependency at serve time).

**② "Your macro-F1 is 0.54. Is that good?"**

The wrong answer is a number. The right answer is that macro-F1 is an *average over classes* and
therefore hides exactly what matters: four of five aspects score 0.000 on neutral, and the headline
number is carried by `absent` (F1 ≈ 0.90–0.98) and `positive`. Then the diagnosis from §6 — that the
neutral collapse tracks support almost perfectly, so it is primarily a data-sparsity problem, while
`misc/negative` at precision 1.000 / recall 0.031 is a separate, fixable calibration problem.
Demonstrating that you know *why* macro-F1 was chosen over accuracy (an all-`absent` predictor scores
~80% accuracy and is worthless) matters more than the value itself.

**③ "Why sqlite? Why not just `mlruns/`?"**

A small question that separates people who followed a tutorial from people who read an error
message. Answer: the file store is in maintenance mode and raises on MLflow 3.x, and it never
supported the Model Registry. Then extend it unprompted to what production looks like — Postgres
backend, S3 artifact store, one-line URI change — which shows you understand the local setup as a
deliberate simplification rather than the only thing you know.

---

## 8. Deliberately not built

Named here so they read as decisions, not omissions:

- **Hyperparameter sweeps / Optuna.** Nothing to sweep yet — the next move is fixing class imbalance,
  which is a loss-function change, not a search.
- **`mlflow.autolog()`.** It hooks the framework and logs a plausible-looking pile of things you did
  not choose. Explicit logging means every recorded value is one I can account for.
- **Nested runs.** Useful for sweeps and cross-validation; both are out of scope.
- **A remote tracking server.** Local sqlite is correct for a single developer.
- **Model stage transitions** (`Staging` → `Production`). Meaningful with a deployment pipeline that
  reads them. We do not have one yet, so it would be theatre.

---

## 9. Files

| File | Role |
|---|---|
| [`ml/src/tracking.py`](../ml/src/tracking.py) | MLflow configuration, per-class metrics, report + confusion-matrix artifacts, model registration. Build-time only. |
| [`ml/src/serving.py`](../ml/src/serving.py) | The pyfunc wrapper. Ships *inside* the artifact — kept deliberately light. |
| [`ml/src/train.py`](../ml/src/train.py) | Now wraps training in an MLflow run; `evaluate()` returns raw predictions; `--eval-test` added. |
| `ml/mlflow.db` | sqlite backend store (gitignored). |
| `ml/mlartifacts/` | Artifact store (gitignored). |

Neither store is committed. Reproducibility comes from committed code plus logged params — not from
committing a 265 MB checkpoint into Git.

**To browse the runs:**

```bash
mlflow ui --backend-store-uri sqlite:///ml/mlflow.db
```

---

## 10. Known gaps carried into Sprint 3

1. **Two load paths** (`predict.load_model` vs the registry) — the API sprint must choose one.
2. **Confusion matrices mislead at low support** — annotate row labels with support counts.
3. **The logged pip requirement for torch is `torch==2.13.0`, not `2.13.0+cu126`.** MLflow strips
   local version labels so the requirement stays installable from PyPI. Harmless here because
   serving is CPU-only, but it means the logged environment does not reproduce the *training* setup
   exactly. Worth knowing before the Docker sprint.
4. **Nothing is tested.** There is still no `pytest` suite. The registry round-trip was verified by
   hand; it should be an automated test.
5. **The real work is still open** — the neutral collapse and `misc/negative` calibration. Sprint 2
   built the instrument; Sprint 3 uses it.
