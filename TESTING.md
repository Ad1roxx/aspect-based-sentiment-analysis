# Manual test checklist

Hand-run checks for the ABSA system.

The automated suite (Sprint 5) covers sections 2, 3 and 5 - run it with `pytest`.
102 tests; `pytest -m "not integration"` runs the 80 that need no trained model.
What remains genuinely manual is section 4: whether the UI actually reads right is
not something a test suite can judge.

Work top to bottom. Anything that fails and is not listed in section 6 is a real bug.

---

## 1. Starting up

```
# terminal 1 - the API
cd e:\absa-project
.venv\Scripts\activate
uvicorn api.main:app --reload --port 8000

# terminal 2 - MLflow UI (optional, for checking runs)
mlflow ui --backend-store-uri sqlite:///ml/mlflow.db --workers 1
```

- [ ] API starts with no traceback
- [ ] Startup log shows: model ready: distilbert-base-uncased (registry version 4)
- [ ] Startup takes a few seconds - that is the 265 MB checkpoint loading, and it
      should happen ONCE at boot, not on the first request
- [ ] http://127.0.0.1:8000/docs renders the interactive Swagger page
- [ ] http://127.0.0.1:8000/ redirects to /docs rather than showing "Not Found"

Expected noise in the uvicorn terminal, not errors:

  DistilBertModel LOAD REPORT ... vocab_transform / vocab_projector UNEXPECTED

  The published distilbert-base-uncased checkpoint carries a masked-language-model
  head. We load AutoModel, the bare encoder, which has no such head, so those five
  tensors are unused. transformers says so itself: "can be ignored when loading
  from different task/architecture".

Expected noise from MLflow, also not errors:

  - "MLflow job execution requirements not met ... does not support Windows" -
    an async job feature we do not use.
  - StarletteDeprecationWarning about starlette.middleware.wsgi - inside MLflow's
    own code, repeated once per worker process.
  - OSError [WinError 10022] with "Child process died" - MLflow starts several
    uvicorn workers that share one socket, and Windows does not inherit sockets
    across processes the way Linux does. The server recovers, but --workers 1
    avoids it entirely. That is why the command above passes it.

If the model is missing you will get "FileNotFoundError ... Train first". Fix with:
python ml/src/train.py --class-weights sqrt-inverse --eval-test --register

---

## 2. API endpoints

**Do not type curl.** FastAPI generates an interactive console at
http://127.0.0.1:8000/docs with a Try it out button for every endpoint, POST included.

  1. open /docs
  2. click the endpoint to expand it
  3. click "Try it out"
  4. edit the request body if there is one
  5. click "Execute"

It shows the status code, the response body, the response headers, and the curl
equivalent if you ever do need one.

The two GET endpoints are also just links you can click straight in the browser:

  http://127.0.0.1:8000/health
  http://127.0.0.1:8000/model-info

Only section 2.7 (CORS) genuinely needs a command line, because a preflight is
something the browser sends, not something you can type into a page.

### 2.1 GET /health

- [ ] Returns 200
- [ ] Body is {"status":"ok","model_loaded":true}
- [ ] model_loaded is true, not merely status ok. A health check that only proves the
      web server answered is decoration

### 2.2 GET /model-info

- [ ] Returns 200
- [ ] registry_version is "4" (not null)
- [ ] run_id, git_commit and trained_at are all populated
- [ ] git_commit matches a real commit - check with: git log --oneline
- [ ] aspects is exactly ["food","service","ambiance","price","misc"] IN THAT ORDER
- [ ] hyperparameters.class_weights is "sqrt-inverse"

This endpoint is the answer to "which model is in production?". If any provenance
field is null, the artifact predates provenance stamping - retrain it.

### 2.3 POST /predict - happy path

In /docs, POST /predict, Try it out, and use this body:

```
{"text": "The pasta was incredible but the waiter ignored us for twenty minutes."}
```

- [ ] Returns 200
- [ ] ALL FIVE aspects are present, including the absent ones. The response shape must
      not change depending on the input
- [ ] food is positive at roughly 71%
- [ ] Each aspect has: aspect, label, confidence, mentioned, words
- [ ] mentioned is false exactly when label is "absent"
- [ ] words is null when explain was not requested
- [ ] latency_ms is present and under about 500 ms

### 2.4 POST /predict - with explanations

Same endpoint, with explain turned on:

```
{"text": "Cosy little place, though it is a bit overpriced for what you get.", "explain": true}
```

- [ ] price is negative at roughly 70%
- [ ] price.words is populated, and "overpriced" has importance 1.000
      (the top word is always exactly 1.0 - scores are normalised)
- [ ] Every importance value is between 0 and 1
- [ ] Aspects scored absent still have words: null - explanations are only computed
      for detected aspects
- [ ] Words are whole words, NOT WordPiece fragments. You should never see "##pric"
- [ ] No [CLS] or [SEP] appears in the word list

### 2.5 Error handling

| Input | Expect |
|---|---|
| text of three spaces | 422, detail "text must contain non-whitespace characters" |
| empty text | 422 |
| missing text field | 422 |
| text as a number | 422 |
| text longer than 5000 chars | 422 |
| GET on /predict | 405 |
| /nope | 404 |

- [ ] All seven behave as listed
- [ ] No error returns a stack trace to the client

### 2.6 Truncation

Send a review longer than 128 tokens (roughly 100+ words):

- [ ] Returns 200 - long input is truncated, not rejected
- [ ] truncated is true
- [ ] This flag exists so the UI can warn. A silently truncated review would produce
      confident predictions about text the model never read

### 2.7 CORS (the React app depends on this)

This one needs a terminal - a preflight is sent by the browser, not typed into a page:

```
curl -i -X OPTIONS http://127.0.0.1:8000/predict -H "Origin: http://localhost:5173" -H "Access-Control-Request-Method: POST"
```

- [ ] Returns 200 with header: access-control-allow-origin: http://localhost:5173
- [ ] The same request with Origin: http://evil.example.com returns NO
      access-control-allow-origin header

---

## 3. Model behaviour spot-checks

These test the model, not the code. Confidence numbers shift if you retrain.

| Review | Expected |
|---|---|
| "The pasta was incredible." | food positive |
| "Service was slow and the waiter was rude." | service negative |
| "It is way too expensive for what you get." | price negative |
| "Lovely warm atmosphere, great for a date." | ambiance positive |
| "I walked past it on my way to work." | everything absent |

- [ ] At least 4 of the 5 behave as expected
- [ ] The last one detects nothing. The model should stay quiet about a review that
      says nothing about the restaurant

Explanation sanity:

- [ ] The top-ranked word is plausibly the sentiment-carrying one
      (incredible, overpriced, rude)
- [ ] Masking that word changes the prediction - replace it with [MASK] and resend

---

## 4. Frontend checks - Sprint 6

The React page does not exist yet. This is the list to work through when it lands.

### 4.1 Layout and content

- [ ] A text area for the review and a clear submit control
- [ ] FIVE aspect cards: food, service, ambiance, price, misc
- [ ] Each card shows the aspect name, the sentiment, and a confidence percentage
- [ ] Sentiment is distinguishable WITHOUT relying on colour alone (icon, or the word
      itself). Red/green alone fails for colour-blind users
- [ ] Confidence renders as a percentage a human reads (71%), not 0.7139

### 4.2 Absent aspects - the thing most likely to be got wrong

- [ ] Aspects the review never mentions are visually de-emphasised, not displayed as a
      confident sentiment
- [ ] "absent at 95%" must never read like a strong opinion. Being 95% sure a topic was
      not discussed is not the same as being 95% sure about the food
- [ ] Decide and check: are absent cards greyed out, or hidden behind a toggle? Either
      is fine. Showing them identically to real sentiments is not

### 4.3 Explanations

- [ ] A toggle or checkbox to request explanations
- [ ] With it off, no highlighting appears and the request is faster
- [ ] With it on, words in the review are highlighted by importance
- [ ] Highlight intensity tracks the importance value
- [ ] Highlighting is PER ASPECT - selecting food vs service shows different words.
      If they look identical, something is wired wrong
- [ ] There is a visible caveat that this shows what the model used, not proof it is
      correct
- [ ] Punctuation sometimes ranks top (see section 6). The UI may filter it, but must
      not misrepresent it

### 4.4 States

- [ ] Loading: a spinner or disabled button while the request is in flight.
      Double-clicking submit must not fire two requests
- [ ] Empty input: submit is disabled, or a friendly message. Never show a raw 422
- [ ] Whitespace-only input: same
- [ ] API down: stop uvicorn, then submit. Expect a readable "cannot reach the service"
      message, not a blank screen or a console-only error
- [ ] Truncated: when the API returns truncated: true, the UI says the review was
      shortened
- [ ] Slow response: the page stays usable and nothing jumps around when results arrive

### 4.5 Basics worth not skipping

- [ ] Works at a narrow window width (about 375 px) - cards stack rather than overflow
- [ ] Keyboard only: Tab to the textarea, type, Tab to submit, Enter works
- [ ] Browser console is clean - no errors, no React key warnings
- [ ] Refreshing mid-use does not leave the page broken
- [ ] No API URL hardcoded to something that only works on your machine

---

## 5. Reproducibility

- [ ] python ml/src/evaluate.py --split test reports overall macro-F1 of 0.6287
- [ ] That matches the training run's test macro_f1 exactly. If it does not, the saved
      artifact is not the model that was measured
- [ ] Retraining with the same seed reproduces the epoch sequence
      0.3379, 0.4644, 0.5342, 0.5395 (with --class-weights none)
- [ ] The MLflow UI lists every run with per-class metrics and confusion matrices

---

## 6. Known issues - do not report these as bugs

Real limitations, measured and documented. Listed here so testing does not rediscover
them as surprises.

1. neutral is broken for four of five aspects. F1 is exactly 0.000 for service,
   ambiance and price. Test supports are 3, 8 and 1 examples - there is nothing there
   to learn from. This is a data problem, not a bug. See explanations/sprint-03.md.

2. service is often missed. "The pasta was incredible but the waiter ignored us"
   returns only food. On "our waiter was incredibly rude" it has returned
   service positive at 40%. Wrong, but with appropriately low confidence.

3. Punctuation can rank first in explanations. The final full stop often takes top
   importance, because tokens next to [SEP] accumulate attribution. It is not filtered
   out, because hiding it would misrepresent the model.

4. Explanation importance is unsigned. It says a word mattered, not which direction it
   pushed. The sign was measured to be untrustworthy and deliberately dropped.

5. Confidence is not calibrated. It is a softmax preference among four options, not a
   probability of being correct. Neural classifiers are typically overconfident.

6. ambiance/negative regressed with class weighting (0.421 to 0.357). A known cost of
   rebalancing, reported rather than hidden.

7. CPU inference is about 200 ms per request. Fine for one review at a time; this is
   not a throughput-optimised service.
