/**
 * One aspect's result.
 *
 * Two decisions here are deliberate and worth keeping:
 *
 * 1. Sentiment is never signalled by colour alone. Each card carries an icon
 *    AND the sentiment word, because red/green alone is unreadable for the
 *    ~8% of men with red-green colour blindness.
 *
 * 2. An `absent` card is visually de-emphasised rather than styled like a
 *    result. "absent at 95%" is 95% confidence that a topic was NOT discussed,
 *    which must not read like a strong opinion about the food.
 */

const SENTIMENT = {
  positive: { icon: "▲", label: "Positive", className: "positive" },
  negative: { icon: "▼", label: "Negative", className: "negative" },
  neutral: { icon: "■", label: "Neutral", className: "neutral" },
  absent: { icon: "–", label: "Not discussed", className: "absent" },
};

// Below this, a "not discussed" call is closer to a coin flip than a finding.
// Measured: a genuinely absent aspect sits at 0.93-0.96, while an aspect the
// model has LOST in a multi-aspect sentence sits around 0.46-0.52.
const UNCERTAIN_ABSENT = 0.7;

export default function AspectCard({ aspect, selected, explainable, onSelect }) {
  const sentiment = SENTIMENT[aspect.label] ?? SENTIMENT.absent;
  const percent = Math.round(aspect.confidence * 100);
  const uncertain = !aspect.mentioned && aspect.confidence < UNCERTAIN_ABSENT;

  const interactive = explainable && aspect.mentioned;
  const classes = [
    "aspect-card",
    sentiment.className,
    selected ? "selected" : "",
    interactive ? "interactive" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <li>
      <button
        type="button"
        className={classes}
        onClick={interactive ? () => onSelect(aspect.aspect) : undefined}
        disabled={!interactive}
        aria-pressed={interactive ? selected : undefined}
      >
        <span className="aspect-name">{aspect.aspect}</span>

        <span className="aspect-sentiment">
          <span aria-hidden="true" className="aspect-icon">
            {sentiment.icon}
          </span>
          {sentiment.label}
        </span>

        <span className="aspect-confidence">
          <span className="confidence-bar" aria-hidden="true">
            <span className="confidence-fill" style={{ width: `${percent}%` }} />
          </span>
          {percent}% confident
        </span>

        {uncertain && (
          <span className="aspect-note">
            low confidence — this aspect may have been missed
          </span>
        )}

        {interactive && (
          <span className="aspect-action">
            {selected ? "showing why" : "show why"}
          </span>
        )}
      </button>
    </li>
  );
}
