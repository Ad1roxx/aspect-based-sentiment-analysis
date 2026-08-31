# Frontend

React + Vite page for the ABSA API.

```bash
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # -> dist/
npm run lint
```

The API must be running separately:

```bash
uvicorn api.main:app --reload --port 8000
```

`VITE_API_URL` selects the API host (see `.env.example`). It is read at **build**
time, not runtime.

## Layout

| File | Role |
|---|---|
| `src/api.js` | fetch client; distinguishes "server unreachable" from a 4xx/5xx |
| `src/components/AspectCard.jsx` | one aspect; icon + word + confidence, never colour alone |
| `src/components/Explanation.jsx` | per-aspect word highlighting |
| `src/App.jsx` | page state, request lifecycle, the five states |

## Two decisions worth knowing

**`absent` cards are de-emphasised, not styled as results.** "absent at 95%" means
95% confidence a topic was *not discussed*; rendered like a sentiment it would read
as a strong opinion about the food.

**A low-confidence `absent` is flagged.** A genuinely absent aspect sits at
0.93–0.96; an aspect the model *lost* in a multi-aspect sentence sits near
0.46–0.52. Below 0.70 the card says so rather than silently hiding a miss.

Manual test checklist: [`../TESTING.md`](../TESTING.md) section 4.
