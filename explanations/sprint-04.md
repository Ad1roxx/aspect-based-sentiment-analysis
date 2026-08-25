# Sprint 4 — FastAPI serving layer

**Goal:** put the model behind an HTTP API — `/health`, `/model-info`, `/predict` — and settle the
two-load-paths question that has been open since Sprint 2.

**Status:** done. All three endpoints working, model loads once at startup, ~200 ms per CPU request.
Manual checklist in [`TESTING.md`](../TESTING.md).

---

## 1. The decision this sprint existed to make

Sprints 2 and 3 both ended with the same unresolved item: there were two ways to load the model, and
an API that consumed neither of them yet.

```
A.  from disk       predict.load_model()  reads ml/models/absa-distilbert/
B.  from registry   mlflow.pyfunc.load_model("models:/absa-distilbert/4")
```

**The service uses A.** Three reasons, in order of weight:

**Serving must not depend on MLflow being reachable.** Option B makes the tracking database a
runtime dependency of the prediction endpoint. If `ml/mlflow.db` is missing, or the artifact store
moved, or the registry is on another host that is down, the API will not start. A prediction service
failing because an *experiment-tracking* system is unavailable is a bad coupling — those two systems
have completely different uptime requirements.

**The container gets smaller and starts faster.** mlflow pulls in Flask, Alembic, SQLAlchemy,
GraphQL, Docker and matplotlib. None of them serve a request. That matters directly in Sprint 7.

**It keeps one code path shared with evaluation.** `predict.load_model()` is the same function
`evaluate.py` uses. Anything that would break serving breaks evaluation first, where it is cheaper
and faster to notice.

### What the registry is still for

It is not abandoned — it becomes the **build-time** source of truth. The intended pipeline: choose a
version in the registry, fetch it during the Docker build, bake it into the image. What the running
service must never do is resolve a model over the network while handling a request.

### The cost of choosing A, and how it was paid

Option B has one genuine advantage: the URI names the version, so you always know what is deployed.
Choosing A gives that up — a directory of files is anonymous.

So the artifact was made to identify itself. `train.py` now stamps four provenance fields into
`metadata.json`:

| field | why |
|---|---|
| `registry_version` | which registry version this artifact is, when promoted |
| `run_id` | the MLflow run — links back to metrics, confusion matrices, params |
| `git_commit` | the code it was trained from |
| `trained_at` | UTC timestamp |

`git_commit` is the one people forget. Params record the *hyperparameters*; the commit records the
*code*. Two runs with identical params still differ if the loss function changed between them, and
nothing else in the metadata catches that.

`registry_version` has to be written *after* `log_model` returns, because the version number does
not exist until the registry assigns it. Hence `stamp_registry_version()` re-opening the file rather
than a single write.

The result is that `/model-info` answers "which model is in production?" from the running service
alone:

```json
{
  "encoder": "distilbert-base-uncased",
  "registry_version": "4",
  "run_id": "8c801269c9ae437eb9040a4f025f0156",
  "git_commit": "2503674988ddf18c028aabbb6fef2c02a057eaad",
  "trained_at": "2026-08-25T16:08:20+00:00"
}
```

---

## 2. API design decisions worth defending

### The model loads at startup, not on first request

Loading happens in FastAPI's `lifespan` handler. A DistilBERT checkpoint is ~265 MB and takes
seconds to read; doing that inside a request handler would make the first prediction unusably slow
and, under concurrency, could start several loads at once.

The subtler benefit: the process now either **boots with a working model or fails loudly**. That is
what you want from a container — a crash-looping pod is obvious, whereas a pod that accepts traffic
and returns 500s looks healthy to everything except its users.

### `/health` reports the model, not the web server

```python
return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)
```

A health check that only proves the HTTP server answered is decoration. It would report "ok" for a
process with no model, which is precisely the failure a health check exists to catch. This is a
small thing that interviewers notice.

### `/predict` always returns all five aspects

Including the ones scored `absent`. The alternative — omitting them — makes the response shape
depend on the input, so the frontend cannot tell "not discussed" from "the API changed" without
extra logic. A fixed shape is easier to consume and easier to type.

`mentioned` is provided as a derived boolean so the UI never has to hard-code the string `"absent"`.

### Explanations are opt-in, and only for detected aspects

Each explanation costs a **separate backward pass**, because the gradient is taken from one aspect's
logit (that is what makes it per-aspect at all — see Sprint 3 §5). Explaining all five aspects would
be five times the work, and four of those results would say "here is why the model thinks price was
never discussed" — true, and almost never what anyone wanted.

So: `explain` defaults to `false`, and when true, only aspects with `mentioned == true` get
attributions. Measured cost on the demo sentences: ~185–215 ms either way, since a one-sentence
forward pass on CPU is dominated by overhead rather than arithmetic.

### `truncated` is surfaced, not hidden

`MAX_LENGTH` is 128 tokens. Longer reviews are truncated by the tokenizer — silently, by default.
That is dangerous: the API would return confident predictions about text the model never read.

The response carries a `truncated` boolean so the UI can say so. Note it counts tokens *with*
special tokens, since `[CLS]` and `[SEP]` occupy two of the 128 positions.

### CORS lists explicit origins

`allow_origins=["http://localhost:5173", ...]` rather than `["*"]`. A wildcard works today and
becomes a finding in any security review. The Vite (5173) and CRA (3000) dev ports are listed
because Sprint 6 needs them.

Verified both directions: an allowed origin gets `access-control-allow-origin` back; a disallowed
one gets a 400 with no such header, so the browser blocks it.

---

## 3. A Pydantic v2 gotcha worth knowing

Pydantic v2 reserves the `model_` prefix for its own attributes and emits a warning for any field
using it. Two fields here are genuinely about the ML model — `model_loaded` on the health response,
and the whole of `ModelInfoResponse`.

The fix is to disable the protection explicitly rather than rename fields into something evasive
like `mdl_loaded`:

```python
ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())

class HealthResponse(BaseModel):
    model_config = ALLOW_MODEL_PREFIX
    ...
```

Renaming would have let Pydantic's internals dictate the public API contract, which is the wrong way
round.

---

## 4. Why `ml/src` goes on `sys.path`

`ml/src` uses flat imports (`from data import ...`) rather than being an installable package, so
`api/service.py` inserts it into `sys.path` instead of importing `ml.src.model`.

This is a compromise and worth naming as one. The clean alternative is to make `ml` a proper package
with `__init__.py` files and relative imports — but that means rewriting every import in every
training module for the benefit of one consumer. The `sys.path` insert is three lines and confined
to a single file.

If the project grew a second consumer of `ml/src`, packaging would become the right call. One
consumer does not justify it yet.

---

## 5. Measured behaviour

| Check | Result |
|---|---|
| `GET /health` | `200` `{"status":"ok","model_loaded":true}` |
| `GET /model-info` | `200`, all provenance fields populated |
| `POST /predict` | `200`, ~214 ms, five aspects |
| `POST /predict` with `explain` | `200`, ~185 ms, `overpriced` at importance 1.000 |
| whitespace-only text | `422` with a specific message |
| empty / missing / wrong-type text | `422` |
| text over 5000 chars | `422` |
| `GET /predict` | `405` |
| unknown route | `404` |
| 200-word review | `200` with `truncated: true` |
| CORS preflight, allowed origin | `200` + `access-control-allow-origin` |
| CORS preflight, other origin | `400`, no allow-origin header |

Note `Field(min_length=1)` accepts `"   "` — three spaces has length 3. The handler strips and
re-checks, so whitespace-only input is a clean 422 rather than a confident prediction about nothing.
Validation at the schema layer and validation at the semantic layer are not the same job.

---

## 6. The three things an interviewer is most likely to probe

**① "You use MLflow's Model Registry. Why doesn't your API load from it?"**

This is the sprint's real question and the answer should be immediate, not improvised: because
serving and experiment tracking have different uptime requirements, and coupling them means the
prediction endpoint dies when the tracking database does. The registry is the build-time source of
truth; the image bakes in the chosen version. Then close the loop unprompted — "the cost is that a
directory of files is anonymous, so the artifact stamps its own run id, git commit and registry
version, and `/model-info` returns them." Naming the trade-off *and* how you paid for it is what
separates a decision from a default.

**② "Walk me through what happens when a request arrives."**

Have the whole path ready: FastAPI validates against the Pydantic schema (422 on failure) → handler
strips and re-checks for whitespace → `service.predict()` counts tokens for the truncation flag →
one forward pass produces `(1, 5, 4)` logits → softmax gives label and confidence per aspect → if
`explain`, one *additional backward pass per detected aspect* from that aspect's logit → response.
The detail that shows real understanding is why explanation cost scales with detected aspects rather
than being free: the gradient must come from a single aspect's logit, or it would not be per-aspect
at all.

**③ "How would you deploy this, and what would you change?"**

Docker is Sprint 7, so answer the *shape*: the CPU-only torch wheel (the pinned `+cu126` build is
~2.5 GB and pointless for serving), the model fetched from the registry at build time, uvicorn behind
multiple workers. Then be honest about what is missing — no auth, no rate limiting, no request
logging or metrics, single-item requests only. Naming the gaps yourself reads as judgement;
being told about them reads as blind spots.

---

## 7. Deliberately not built

- **Batch endpoint.** `/predict` takes one review. Batching is genuinely useful and genuinely more
  API surface (partial failures, size limits, response ordering). Out of scope.
- **Auth / rate limiting.** Named as out of scope in the project brief.
- **Async handlers.** The routes are `def`, not `async def`, so FastAPI runs them in a threadpool.
  That is *correct* here: torch inference is blocking CPU work, and `async def` would block the event
  loop for 200 ms per request, which is worse.
- **Streaming / websockets.** Nothing to stream — one prediction, one response.
- **Structured request logging and metrics.** Real production needs both. Listed as future work.

---

## 8. Files

| File | Role |
|---|---|
| [`api/schemas.py`](../api/schemas.py) | Pydantic request/response contract; also generates the OpenAPI docs |
| [`api/service.py`](../api/service.py) | Loads the model once; wraps prediction and explanation |
| [`api/main.py`](../api/main.py) | Routes, lifespan startup, CORS |
| [`TESTING.md`](../TESTING.md) | 67-item manual checklist, incl. the frontend list for Sprint 6 |
| [`ml/src/train.py`](../ml/src/train.py) | Now stamps provenance into `metadata.json` |

```
uvicorn api.main:app --reload --port 8000
open http://127.0.0.1:8000/docs
```

---

## 9. Carried into Sprint 5 (tests)

1. **Still zero automated tests.** Everything above was verified by hand. `TESTING.md` is the
   specification to automate against — the error-code table especially.
2. **Tests need a model artifact**, which is 265 MB and gitignored. CI has no GPU and should not
   train. This is a real problem to solve: either a tiny fixture model, or mocking the service layer,
   or training a 1-epoch model in CI. Decide deliberately.
3. **Three hand-verified behaviours should become test cases**: the registry round-trip (Sprint 2),
   save/load fidelity — `evaluate.py` reproducing 0.6287 exactly (Sprint 3) — and the occlusion check
   on explanations (Sprint 3).
4. **`latency_ms` is measured but not recorded anywhere.** Fine for now; it is the natural hook for
   metrics later.
