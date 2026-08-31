# Sprint 9 — Docker and CI/CD

**Goal:** package the API and frontend as containers, and have GitHub run the tests on every push.

**Status:** done. `docker compose up` serves both. 147 tests pass. Two real bugs found — one that
*only* a container could have surfaced.

---

## 1. The dependency split

The serving path imports exactly four third-party packages: `fastapi`, `pydantic`, `torch`,
`transformers`. Nothing else is reachable from `api/main.py`. So `requirements-serve.txt` contains
only those, and the difference is not marginal:

| left out | size |
|---|---|
| **torch `+cu126` to CPU** | **8,451 MB** |
| scikit-learn + scipy | 345 MB |
| mlflow + Flask/Alembic/SQLAlchemy/GraphQL | 195 MB |
| pandas | 140 MB |
| matplotlib | 66 MB |

The CUDA build ships GPU kernels a CPU-only service never calls — inference runs on CPU by design
(sprint 4), because a single short sentence is dominated by overhead rather than matrix
multiplication.

**`pip install torch` gets you the CUDA build from PyPI.** The CPU wheel lives on PyTorch's own
index and must be requested explicitly, which is why the Dockerfile installs it as a separate step.
Separate also means the largest layer caches independently — editing `requirements-serve.txt` does
not re-download 1.5 GB.

**Result: 2.23 GB, including the 254 MB model.** I estimated ~1.2 GB beforehand and was wrong; CPU
torch is still large. But against ~9.5 GB naively, it is the difference between an image you can push
and one you cannot.

---

## 2. The bug only a container could find

The API image sets `HF_HUB_OFFLINE=1`, on the reasoning that the tokenizer and weights are baked in
so the container should need no network. It crash-looped:

```
OSError: We couldn't connect to 'https://huggingface.co' to load the files
  File "/app/ml/src/model.py", line 94, in __init__
    self.encoder = AutoModel.from_pretrained(encoder_name)
```

**`AutoModel.from_pretrained` downloads the base encoder — and `load_state_dict` overwrites every
one of those weights on the very next line.** The artifact had never been self-contained. Since
sprint 1 it had been fetching 250 MB over the network purely to obtain an *architecture*, then
throwing the weights away.

Nothing caught it earlier because every machine that ran it had either a network or a populated
`HF_HOME` cache. It is exactly the class of "works on my machine" that containers exist to expose.

The fix is small and correct: `save_artifact` now writes the encoder's `config.json` beside the
weights, and loading builds the architecture from that. Training still uses `from_pretrained` —
fine-tuning genuinely needs pretrained weights; loading a finished artifact does not.

Verified by loading with `HF_HOME` pointed at a nonexistent path and both offline flags set: it
works, and still returns `ambience positive` / `price negative`.

**~1 KB of JSON removed a 250 MB network dependency.**

---

## 3. The bug curl could never find

The containerised page is served by nginx on **:8080**. The API's CORS allow-list had the Vite dev
server (5173) and CRA (3000) — not 8080.

```
OPTIONS /predict   Origin: http://localhost:8080   ->   400, no allow-origin header
```

**A terminal test cannot catch this.** curl does not enforce CORS; it happily returns a body the
browser would have refused. The page would have loaded perfectly and every prediction failed, with
the only evidence in the browser console.

`ALLOWED_ORIGINS` is now an environment variable with sensible defaults, because the correct value is
a deployment fact rather than a code fact. There is now a test for the `:8080` origin specifically.

---

## 4. The frontend gotcha, avoided rather than discovered

Vite inlines `import.meta.env` at **build** time, so the API URL is compiled into the JavaScript and
cannot be changed by setting an environment variable on the running container. It is a `--build-arg`.

The subtler half: it must be an address **the browser** can reach. The browser runs on the host,
outside the compose network, so `http://api:8000` — which is perfectly valid *between* containers —
resolves to nothing in the page. It builds clean, starts clean, and fails only on click.

Both the compose file and the Dockerfile carry a comment saying why.

---

## 5. `.dockerignore` is not optional here

Without it the build context is roughly 10 GB: the 8.4 GB `.venv`, `node_modules`, MLflow artifacts,
and the raw corpora. Docker copies the context *before* it starts building, so every build would
stall before running a single instruction.

---

## 6. CI — three jobs

```
test       124 tests (pytest -m "not integration")
frontend   npm ci -> oxlint -> npm run build
docker     both images build
```

The test job is the payoff for a decision made four sprints earlier. Sprint 5 layered the suite so
that pure-logic and API tests need no GPU, no network and no artifact — and the 22 integration tests
mark themselves and **skip** when `ml/models/` is empty, which on a fresh clone it always is. No
special CI configuration was needed; the suite was already shaped for it.

One CI-specific wrinkle: `requirements.txt` pins `torch==2.13.0+cu126`. Asking a GitHub runner to
download 8.4 GB of CUDA wheels would be absurd, so the workflow installs the CPU wheel first and then
strips the torch pin from the requirements file before installing the rest.

**The docker job has an honest limit**, stated in the workflow: it proves the images *build*, not
that they serve. The 254 MB artifact is gitignored, so the API image built in CI contains no model
and the container would exit at startup. An `ml/models/.gitkeep` is committed so the `COPY` has
something to copy on a fresh clone.

---

## 7. CD — where I stopped pretending

`release.yml` publishes both images to the GitHub Container Registry when a version tag is pushed.
After it runs, anyone can pull `ghcr.io/<owner>/...-api:v1.0.0`.

That is **not continuous deployment**, and the workflow says so in a comment. There is no server to
deploy to. A pipeline that ran `echo "deploying..."` and exited 0 would look like CD in a README and
be worth nothing — and it is the kind of claim an interviewer pokes at first.

Note the published API image contains **no model**. Separating application from weights is normal for
model serving, and it is why the artifact carries its own provenance (sprint 4): the image is
versioned by tag, the model by registry version, and `/model-info` reports which is running.

---

## 8. The three things an interviewer is most likely to probe

**(1) "Why is your image 2.2 GB?"**

Because torch is. Lead with what you *removed* and why you know it was safe: the serving path imports
four packages, so mlflow, sklearn, matplotlib and pandas are absent, and the CUDA build is replaced
with the CPU wheel — 8.4 GB of GPU kernels a CPU-only service never calls. Then be honest that 2.2 GB
is still large, and name the next lever if pressed: ONNX Runtime or a distilled export would drop the
torch dependency entirely.

**(2) "What did containerising actually catch?"**

The strongest answer in the sprint, because it is specific. `from_pretrained` was downloading 250 MB
of pretrained weights that `load_state_dict` overwrote on the next line — the artifact had never been
self-contained, and no machine with a network or a warm cache would ever have revealed it. Then the
CORS one, whose lesson is different and just as useful: **curl does not enforce CORS**, so the
failure was invisible to every terminal test and would have appeared only in the browser.

**(3) "Your CI does not deploy. Why call it CD?"**

Do not defend it — agree. There is no host; publishing versioned images to a registry is the real
part, and calling it deployment would be theatre. Then show you know what the missing piece is: a
target, a secret, and a step that pulls the tagged image. That answer is stronger than a fake deploy
job, because interviewers have seen plenty of those.

---

## 9. Files

| File | Role |
|---|---|
| [`requirements-serve.txt`](../requirements-serve.txt) | serving deps only; torch installed separately |
| [`Dockerfile.api`](../Dockerfile.api) | two-stage build, non-root, model-aware healthcheck |
| [`frontend/Dockerfile`](../frontend/Dockerfile) | node build, then nginx |
| [`docker-compose.yml`](../docker-compose.yml) | both services; `web` waits for `api` to be *healthy* |
| [`.dockerignore`](../.dockerignore) | keeps the context from being ~10 GB |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | tests, frontend, image builds |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml) | publish to GHCR on a tag |

```bash
docker compose up --build
#   API   http://localhost:8000/docs
#   page  http://localhost:8080
```

---

## 10. Carried into Sprint 10 (docs)

1. **README is still the Sprint 1 stub** — needs the architecture diagram, results, and honest
   limitations. That is the last item in the definition of done.
2. **CI has never actually run.** It is committed but unverified until the next push, and the first
   run will probably need a fix. Normal, and worth saying rather than claiming green.
3. **The image contains a model CI cannot reproduce.** Worth a README paragraph on how someone else
   gets a working container: train locally, or pull a registry version.
