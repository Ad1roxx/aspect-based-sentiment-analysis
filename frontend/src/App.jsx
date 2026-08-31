import { useEffect, useRef, useState } from "react";
import AspectCard from "./components/AspectCard";
import Explanation from "./components/Explanation";
import { ApiError, MAX_LENGTH, modelInfo, predict } from "./api";
import "./App.css";

const EXAMPLES = [
  "The pasta was incredible but the waiter ignored us for twenty minutes.",
  "Cozy little place, though it is a bit overpriced for what you get.",
  "I walked past it on my way to work.",
];

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [explain, setExplain] = useState(true);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState(null);

  // Ignore a response if a newer request has already been sent. Without this,
  // a slow first request can land after a fast second one and overwrite it.
  const requestId = useRef(0);

  useEffect(() => {
    modelInfo()
      .then(setInfo)
      .catch(() => setInfo(null)); // the banner is optional; never block the page
  }, []);

  const trimmed = text.trim();
  const tooLong = text.length > MAX_LENGTH;
  // Guards the empty AND whitespace-only cases in one place, so the button is
  // simply unavailable rather than the user meeting a raw 422.
  const canSubmit = trimmed.length > 0 && !tooLong && !loading;

  async function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) return;

    const id = ++requestId.current;
    setLoading(true);
    setError(null);

    try {
      const data = await predict(trimmed, explain);
      if (id !== requestId.current) return;
      setResult(data);
      // Preselect the first detected aspect so an explanation is visible
      // immediately rather than requiring a second click to see anything.
      setSelected(data.aspects.find((a) => a.mentioned)?.aspect ?? null);
    } catch (caught) {
      if (id !== requestId.current) return;
      setError(
        caught instanceof ApiError ? caught.message : "Something went wrong.",
      );
      setResult(null);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }

  const selectedAspect = result?.aspects.find((a) => a.aspect === selected);
  const detected = result?.aspects.filter((a) => a.mentioned) ?? [];

  return (
    <main className="page">
      <header>
        <h1>Aspect-Based Sentiment Analysis</h1>
        <p className="subtitle">
          Restaurant reviews, scored across five aspects.
        </p>
        {info && (
          <p className="model-banner">
            {info.encoder} · registry v{info.registry_version} · validation
            macro-F1 {info.validation_metrics?.macro_f1}
          </p>
        )}
      </header>

      <form onSubmit={handleSubmit}>
        <label htmlFor="review">Review</label>
        <textarea
          id="review"
          value={text}
          rows={4}
          maxLength={MAX_LENGTH + 1}
          placeholder="Paste a restaurant review…"
          onChange={(event) => setText(event.target.value)}
        />

        <div className="examples">
          <span>Try:</span>
          {EXAMPLES.map((example, index) => (
            <button
              key={example}
              type="button"
              className="link"
              onClick={() => setText(example)}
            >
              example {index + 1}
            </button>
          ))}
        </div>

        <div className="controls">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={explain}
              onChange={(event) => setExplain(event.target.checked)}
            />
            Explain predictions
            <span className="hint">(slower — one extra pass per aspect)</span>
          </label>

          <button type="submit" disabled={!canSubmit}>
            {loading ? "Analysing…" : "Analyse"}
          </button>
        </div>

        {tooLong && (
          <p className="inline-warning">
            That is {text.length.toLocaleString()} characters; the limit is{" "}
            {MAX_LENGTH.toLocaleString()}.
          </p>
        )}
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {/* Reserve the results area so nothing below jumps when a result lands. */}
      <div className="results" aria-busy={loading}>
        {result && (
          <>
            {result.truncated && (
              <p className="warning" role="status">
                This review was longer than the model reads (128 tokens). Only
                the beginning was analysed.
              </p>
            )}

            <ul className="aspect-grid">
              {result.aspects.map((aspect) => (
                <AspectCard
                  key={aspect.aspect}
                  aspect={aspect}
                  selected={aspect.aspect === selected}
                  explainable={result.explained}
                  onSelect={setSelected}
                />
              ))}
            </ul>

            {detected.length === 0 && (
              <p className="empty-note">
                No aspects detected — this review does not appear to discuss the
                restaurant itself.
              </p>
            )}

            {result.explained && selectedAspect?.words && (
              <Explanation
                aspect={selectedAspect.aspect}
                words={selectedAspect.words}
              />
            )}

            <p className="latency">Responded in {result.latency_ms} ms</p>
          </>
        )}
      </div>
    </main>
  );
}
