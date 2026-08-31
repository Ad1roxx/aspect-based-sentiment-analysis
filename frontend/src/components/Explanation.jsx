/**
 * Per-word importance for ONE aspect.
 *
 * The highlighting is per aspect on purpose: the API returns a different word
 * list for each, because the attribution is the gradient of that aspect's own
 * logit. If food and service ever rendered identically, something would be
 * wired wrong — that is the exact failure mode attention-based explanation
 * would have had with the old shared-[CLS] architecture.
 *
 * Importance is UNSIGNED. It says a word mattered, not which way it pushed the
 * prediction, and the caption says so rather than letting the colour imply a
 * direction it cannot support.
 */

// Words below this are visually silent. Attribution has a long tail of tiny
// values, and shading every token makes the page look uniformly grey.
const FLOOR = 0.15;

export default function Explanation({ aspect, words }) {
  if (!words?.length) return null;

  const strongest = words.reduce((a, w) => (w.importance > a ? w.importance : a), 0);
  const topWord = words.reduce((a, w) => (w.importance > a.importance ? w : a), words[0]);

  return (
    <section className="explanation" aria-live="polite">
      <h3>
        Why <em>{aspect}</em>?
      </h3>

      <p className="explanation-text">
        {words.map((word, index) => {
          const relative = strongest > 0 ? word.importance / strongest : 0;
          const visible = relative >= FLOOR;
          return (
            <span
              key={`${word.word}-${index}`}
              className={visible ? "token highlighted" : "token"}
              style={visible ? { opacity: 0.25 + relative * 0.75 } : undefined}
              title={`importance ${word.importance.toFixed(3)}`}
            >
              {word.word}
            </span>
          );
        })}
      </p>

      <p className="explanation-caption">
        Strongest signal: <strong>{topWord.word}</strong>. Shading shows which
        words the model relied on, not whether they were positive or negative —
        the direction is not something this method can tell you reliably.
        Punctuation sometimes ranks highly; that is a known artefact.
      </p>
    </section>
  );
}
