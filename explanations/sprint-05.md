# Sprint 5 — Tests

**Goal:** a pytest suite that runs without a GPU, without the 265 MB artifact, and without network —
plus integration tests that use the real model when it is present.

**Status:** done. **102 tests passing.** 80 run with no artifact at all (the CI path); 22 are
integration tests that skip cleanly when `ml/models/` is empty. `api/` is at 98–100% coverage.

---

## 1. The constraint that shaped everything

Sprint 4 ended with a problem, not a plan:

> Tests need a model artifact, which is 265 MB and gitignored, and CI has no GPU.

Three bad answers were available:

- **Commit the artifact.** 265 MB in Git, and Git stores every version forever. No.
- **Train in CI.** Downloads the dataset, needs a GPU to be quick, and makes every push wait
  ~50 seconds — on a machine with no GPU, minutes. It also means a CI failure could be a *training*
  failure, which is a terrible signal to debug.
- **Skip testing the model.** Then "we have tests" is technically true and practically worthless.

The answer is that **most of what needs testing does not involve the model at all**. Once you notice
that, the suite designs itself into three layers:

| Layer | Needs | Count | Runtime |
|---|---|---|---|
| **Pure logic** — parsing, label mapping, loss masking, class weights, metrics, subword merging | nothing | 54 | ~0.1 s (plus torch import) |
| **API contract** — routes, validation, status codes, CORS, response shape | a *fake* service | 26 | ~1 s |
| **Integration** — real artifact, real predictions, explanations, registry | the 265 MB artifact | 22 | ~35 s |

```bash
pytest                        # everything (102)
pytest -m "not integration"   # the CI path (80, no artifact needed)
pytest -m integration         # just the model tests
```

The layering is not merely a convenience. A test that loads a 265 MB checkpoint to check that empty
input returns 422 is testing the wrong thing anyway — it is slower, it is harder to debug when it
fails, and it conflates "the web layer is correct" with "the model exists".

### How the skipping works

```python
artifact_required = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.pt").is_file(),
    reason=f"no trained artifact at {ARTIFACT_DIR} — run ml/src/train.py",
)
```

Verified by actually moving `model.pt` away and re-running: all 22 integration tests **skip** with
that message rather than erroring. That check matters — a `skipif` that raises during collection
instead of skipping would break CI exactly as badly as no guard at all.

`--strict-markers` is set in `pyproject.toml`, so a typo'd marker becomes an error instead of
silently running the test unmarked. That is how "we have integration tests" quietly becomes false.

### `pythonpath` instead of `sys.path` hacks

`ml/src` uses flat imports, so tests need it importable. pytest's `pythonpath` ini option handles it
for the whole session:

```toml
pythonpath = ["ml/src", "."]
```

No `sys.path.insert` scattered through `conftest.py`.

---

## 2. Testing the API without a model

The trick is that `api/main.py` holds a reference to a module-level `service` object, so patching
attributes on that object swaps the model out while leaving every route exactly as written:

```python
monkeypatch.setattr(svc, "load", lambda: None)
monkeypatch.setattr(svc, "model", object())      # makes is_loaded True
monkeypatch.setattr(svc, "metadata", FAKE_METADATA)
monkeypatch.setattr(svc, "predict", fake_predict)
```

No dependency-injection scaffolding was added to the app purely to make it testable. That matters:
production code contorted for tests is a cost paid forever, and FastAPI's `Depends` would have been
exactly that here.

This layer also tests the one state that is otherwise hard to reach — a server that **started but
has no model**. That is the whole reason `/health` reports on the model rather than on the web
server, and there is now a test that fails if someone "simplifies" it to `return {"status": "ok"}`.

---

## 3. Four things writing the tests actually found

Tests are supposed to find things. These are what turned up.

### ① Weighted cross-entropy cancels with a single example

Writing a test called `test_upweighting_a_class_raises_its_cost`, the assertion failed. The reason is
real and non-obvious: `F.cross_entropy` with weights computes

```
sum(w_i * loss_i) / sum(w_i)
```

— it normalises by the **summed weights of the targets actually present**. With one example, the
weight appears in numerator and denominator and cancels *exactly*. The class weight has no
observable effect at all.

It only shows up with a batch of differently-labelled examples, where upweighting one class shifts
the average toward it. The test now uses two examples — one `absent` predicted correctly, one
`negative` predicted wrongly — and asserts the batch loss rises.

This is a genuine trap for anyone hand-verifying class weighting: test it on one example and you
conclude, wrongly, that your weights are not being applied.

### ② A test whose name overstated what it checked

The first version of that test, on failing, was quietly weakened to assert only that the loss was
finite — while keeping the name `..._raises_its_cost`. That is worse than no test: it reports
coverage of a behaviour nobody is checking. It was rewritten to actually demonstrate the effect.

Worth internalising as a habit: when a test fails, the options are *fix the code* or *fix your
understanding*. Weakening the assertion until it passes is neither.

### ③ A deprecation warning in our own code

The suite surfaced `HTTP_422_UNPROCESSABLE_ENTITY is deprecated`. Not a library's problem — ours, in
`api/main.py`. Fixed to `HTTP_422_UNPROCESSABLE_CONTENT`. A test suite that runs clean is a suite
where the next warning gets noticed.

### ④ The untested glue between API and model

Coverage showed `api/service.py` at **60%**, with `ModelService.predict` uncovered. The reason was
structural and easy to miss: the API tests replace that method with a fake, and the integration
tests call `model.predict` directly. Both sides were tested; the wiring between them was not — which
is precisely where a real bug hides.

`TestRealModelService` now drives the actual service: truncation counting, the
explain-only-if-mentioned rule, the response dict shape. **60% → 98%.**

Coverage was useful here not as a score but as a *question*: why is this number low? The answer was
a real hole.

---

## 4. A floor, not an exact match

The performance test asserts:

```python
assert metrics["macro_f1"] > 0.55
```

not `== 0.6287`.

An exact assertion would fail the moment the model is **improved**, which is a test that punishes
progress and gets deleted within a week. The floor catches what actually matters: an artifact that
loads but is broken, a retrain that quietly made things worse, a corrupted checkpoint.

The exact-reproduction property is still valuable — it is how Sprint 3 proved the save/load path was
faithful — but it belongs in a manual check ([`TESTING.md`](../TESTING.md) §5), not in a suite that
must stay green across intentional model changes.

Same reasoning for the known-example spot-checks: they deliberately exclude `service`, because the
model is known to miss it (TESTING.md §6). Asserting a behaviour you know to be broken produces a
red suite that everyone learns to ignore.

---

## 5. Coverage, honestly

```
api/main.py         100%
api/schemas.py      100%
api/service.py       98%
ml/src/evaluation.py 100%
ml/src/serving.py    92%
ml/src/data.py        74%
ml/src/model.py       66%
ml/src/explain.py     65%
ml/src/predict.py     57%
ml/src/tracking.py    49%
ml/src/train.py       49%
ml/src/evaluate.py     0%
TOTAL                66%
```

The serving path — the code that actually runs in production — is at 98–100%. The gaps are
concentrated and each has a reason:

- **`evaluate.py` at 0%** — it is a `main()` CLI wrapper around already-tested functions. Testing
  argparse plumbing has poor value per line.
- **`train.py` at 49%** — the uncovered half is the training loop, which cannot run without a GPU and
  a dataset. Its *components* (class weights, loss, evaluation, metadata) are tested individually.
- **`tracking.py` at 49%** — MLflow logging calls. Testing them means asserting that MLflow was
  called, which mostly tests the mock.

66% is a truthful number, not a padded one. It would be easy to reach 85% by testing `main()`
functions and argparse, and none of it would catch a real bug.

---

## 6. The three things an interviewer is most likely to probe

**① "Your model is 265 MB and CI has no GPU. How do you test it?"**

This is the design question of the sprint. The answer is the layering — and specifically the
realisation that *most of what needs testing does not involve the model*. Parsing, label mapping,
loss masking, class weights, metric computation and subword merging are all pure functions.
`compute_loss` takes tensors rather than a model, so the `IGNORE_INDEX` masking — the part most
likely to be subtly wrong — is verified with no checkpoint at all. Then: API tests through a fake,
integration tests marked and skipped. Mention that you verified the skip by moving `model.pt` away,
because a guard that errors during collection would break CI just as badly as no guard.

**② "What did writing the tests find?"**

Have a specific answer ready, because "they all passed" implies the tests were written to match the
code. The best one is the weighted-cross-entropy discovery: with a single example the class weight
cancels exactly, because the loss normalises by the summed weights of present targets. Then the
coverage hole — `api/service.py` at 60% because API tests faked the service and integration tests
bypassed it, leaving the wiring untested. That one shows you read coverage as a question rather than
a score.

**③ "Why is your coverage only 66%?"**

Do not apologise for it. The serving path is 98–100%; the gap is training-loop code that needs a GPU
and CLI wrappers around already-tested functions. Then make the sharper point: you could reach 85%
by testing `argparse`, and it would catch nothing. Coverage measures which lines ran, not whether
they were checked — a test that calls a function and asserts nothing scores identically to a good
one. Being able to say *why* your number is what it is beats a higher number you cannot account for.

---

## 7. Deliberately not built

- **Property-based testing (Hypothesis).** A good fit for `merge_subwords`, genuinely. Deferred as
  scope; the hand-written edge cases cover the failure modes that exist.
- **Snapshot testing of model outputs.** Would break on every retrain, which is the exact brittleness
  §4 avoids.
- **Load / performance tests.** Meaningful once there is a deployment target. Sprint 7 at the earliest.
- **Mutation testing.** The right tool for asking whether tests actually assert anything, and far
  beyond scope here.
- **A tiny randomly-initialised fixture model.** Considered as a fourth layer, so shape tests could
  run without the real artifact. Rejected: `compute_loss` already takes tensors, so the tests that
  would have needed it do not exist.

---

## 8. Files

| File | Contents |
|---|---|
| [`tests/conftest.py`](../tests/conftest.py) | XML fixtures (incl. the phase B trap), sample examples, the `artifact_required` skip |
| [`tests/test_data.py`](../tests/test_data.py) | Parsing, aliases, conflict masking, split determinism, class counts |
| [`tests/test_model.py`](../tests/test_model.py) | Loss at tensor level: ln(4) check, `IGNORE_INDEX` masking, class weights |
| [`tests/test_train.py`](../tests/test_train.py) | Class-weight schemes and their ordering |
| [`tests/test_evaluation.py`](../tests/test_evaluation.py) | Per-class metrics, report text, confusion figure edge cases |
| [`tests/test_explain.py`](../tests/test_explain.py) | Subword merging, special tokens, signed-sum aggregation |
| [`tests/test_api.py`](../tests/test_api.py) | Routes, validation, CORS, OpenAPI — through a fake service |
| [`tests/test_integration.py`](../tests/test_integration.py) | Real artifact: predictions, explanations, occlusion, registry, ModelService |
| [`pyproject.toml`](../pyproject.toml) | pytest + coverage config |
| [`requirements-dev.txt`](../requirements-dev.txt) | Test deps, kept out of the serving image |

---

## 9. Carried into Sprint 6 (frontend)

1. **The API contract is now enforced by tests.** `test_api.py` is the specification React codes
   against — field names, the all-five-aspects rule, `mentioned`, `truncated`. Changing the response
   shape now breaks a test rather than the browser.
2. **CORS is tested for both directions**, so a frontend on `localhost:5173` will work and an
   unlisted origin will not.
3. **`TESTING.md` §4 is the frontend checklist** — 24 items, written in Sprint 4, still unrun.
4. **The 0% on `evaluate.py`** is the one gap I would close first if asked to raise coverage
   honestly — a single smoke test that runs `main()` against a tmpdir.
