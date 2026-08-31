# Sprint 8 — React frontend

**Goal:** a page that consumes `/predict` — aspect cards with sentiment and confidence, per-aspect
explanation highlighting, and the states a real user will hit.

**Status:** done. Vite + React 19, no dependencies beyond `react`/`react-dom`. Lint clean, builds to
62 kB gzipped.

---

## 1. Every design decision here came from a measurement

This is the part worth defending in an interview, because a frontend is usually where projects stop
being evidence-driven and start being taste. Each of these traces back to a number from an earlier
sprint.

### `absent` cards are de-emphasised, not styled as results

The API returns all five aspects, most of them `absent` with confidence around 0.95. Rendered like
the others, a card reading **"price — absent — 95% confident"** looks like a strong claim about
price. It is 95% confidence that price was *never mentioned*, which is a completely different
statement.

Absent cards are therefore flat, muted, at 62% opacity, with no sentiment colour and the label
"Not discussed" rather than the raw class name.

### A low-confidence `absent` is flagged to the user

Sprint 7 measured something the UI can act on:

| | typical confidence |
|---|---|
| a genuinely absent aspect | **0.93 – 0.96** |
| an aspect *lost* in a multi-aspect sentence | **0.46 – 0.52** |

Those distributions barely overlap. So below 0.70 the card adds *"low confidence — this aspect may
have been missed"*. That is the model's own uncertainty surfaced rather than hidden, and it directly
addresses the residual multi-aspect gap (0.064) that no amount of data fixed completely.

Sprint 7 also showed a global confidence *threshold* is a losing trade — it destroys more true
positives than false ones. Showing the uncertainty costs nothing and loses nothing, which is why the
UI does that instead of silently dropping low-confidence calls.

### Sentiment is never signalled by colour alone

Each card carries an icon (▲ ▼ ■ –) **and** the sentiment word, with colour as reinforcement.
Red/green alone is unreadable for roughly 8% of men. This is the cheapest accessibility win
available and it costs one span.

### Highlighting is per aspect, and the caption says what it cannot tell you

Selecting a card shows *that aspect's* attribution. This is only meaningful because Sprint 6 replaced
shared-`[CLS]` pooling — with the old architecture every aspect would have produced an identical
highlight, which looks like an explanation and is not one.

The caption states the importance is **unsigned**: it says a word mattered, not which way it pushed.
Sprint 3 measured that the sign was untrustworthy (`overpriced` drew the largest magnitude with a
*negative* sign on a `price → negative` prediction), so letting the shading imply direction would be
asserting something known to be false. The known punctuation artefact is named too.

### Explanations are opt-in because they cost 9x

Measured on the same warm server:

```
explain=false     21 ms
explain=true     185 ms
```

One extra backward pass per detected aspect. The checkbox says "slower" and defaults on for a demo,
but the cost is real and the API contract already made it optional for this reason.

---

## 2. Request handling, and one bug that would only appear under load

```js
const id = ++requestId.current;
const data = await predict(trimmed, explain);
if (id !== requestId.current) return;   // a newer request has superseded this one
```

Without that guard, submitting twice in quick succession can let the *slower first* response land
after the faster second one and overwrite it — the UI then shows results for text the user has
already replaced. The submit button is also disabled while `loading`, which prevents the common case,
but a disabled button does not help if the user edits and resubmits legitimately.

The results container reserves `min-height: 320px`, so the page does not jump when a result arrives.

**Errors are split by kind.** `fetch` only rejects on network-level failure, so that branch means the
API is unreachable and the user gets *"Cannot reach the API at …. Is the server running?"*. A 4xx/5xx
is a different problem and shows the server's own `detail`, including Pydantic's field errors. A raw
422 is never shown: empty and whitespace-only input simply disable the button.

---

## 3. What I could not verify

**No browser automation was available in this environment, so the page was never visually rendered
during development.** That is a real gap and it is stated rather than glossed over. What *was*
verified automatically:

- production build succeeds (62 kB gzipped, 20 modules)
- `oxlint` clean — no missing keys, unused vars, or hook misuse
- both servers respond (`/health` 200, Vite 200)
- the exact request the page makes returns the expected result end to end
- expected strings are present in the built bundle

[`TESTING.md`](../TESTING.md) section 4 — 24 items covering layout, absent handling, explanations,
states, and accessibility basics — is the manual pass, and it is genuinely manual. Whether the UI
*reads* right is not something a build step can judge.

---

## 4. Files

| File | Role |
|---|---|
| [`frontend/src/api.js`](../frontend/src/api.js) | fetch client; separates "unreachable" from a 4xx/5xx |
| [`frontend/src/components/AspectCard.jsx`](../frontend/src/components/AspectCard.jsx) | one aspect; icon + word + confidence |
| [`frontend/src/components/Explanation.jsx`](../frontend/src/components/Explanation.jsx) | per-aspect word highlighting |
| [`frontend/src/App.jsx`](../frontend/src/App.jsx) | page state and the request lifecycle |

```bash
uvicorn api.main:app --reload --port 8000     # terminal 1
cd frontend && npm install && npm run dev     # terminal 2
```

---

## 5. Carried into Sprint 9 (Docker)

1. **`VITE_API_URL` is read at BUILD time, not runtime.** A container image is therefore pinned to
   whatever the API URL was when `npm run build` ran. Either build per environment, or serve a small
   runtime config file the page fetches. This needs deciding, not discovering.
2. **Two processes to containerise** — uvicorn and a static server for `dist/`. Compose with two
   services, or one image with nginx in front.
3. **The CPU torch wheel.** The pinned `+cu126` build is ~2.5 GB and pointless for serving; the
   requirements split promised back in Sprint 1 lands here.
4. **The model artifact is 265 MB and gitignored** — the image must fetch it from the MLflow registry
   at build time, which is exactly the "registry as build-time source of truth" design from Sprint 4.
