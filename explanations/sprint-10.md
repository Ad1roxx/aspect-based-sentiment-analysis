# Sprint 10 — Documentation

**Goal:** the last item in the definition of done — a README with an architecture diagram, results
and honest limitations.

**Status:** done. The project is complete against its original scope.

---

## 1. What a README is actually for

Someone landing on this repo decides in about thirty seconds whether it is worth reading. So the
order is deliberate:

1. **what it does**, with a concrete example — not a paragraph about ABSA as a field
2. **the architecture diagram**, because one picture replaces three paragraphs
3. **results**, including how they were reached
4. **quick start**, so it can be run before it is understood
5. **decisions**, linking out rather than re-explaining
6. **limitations**, measured

The diagram is a mermaid block, which GitHub renders natively — no image file to regenerate and go
stale, and it stays diffable in review.

---

## 2. The results table earns more than the number

The headline is 0.638 test macro-F1. On its own that is unimpressive and unfalsifiable. What makes it
worth reading is the progression, because each row is a measured experiment:

| | test macro-F1 |
|---|---|
| baseline | 0.550 |
| \+ class-weighted loss | 0.629 |
| \+ attention pooling | 0.632 |
| \+ MAMS data | 0.638 |

And then the point the headline hides — multi-aspect detection went 76.7% → 82.3%, gap 0.125 →
0.064. **Macro-F1 averages that away**, because a lost aspect looks like a correct `absent`
prediction next to four other correct ones. Showing both is the difference between reporting a score
and reporting what the model does.

---

## 3. Writing limitations you would not be embarrassed by

The temptation is to hedge — "performance on neutral could be improved". That reads as either
ignorance or evasion. Every limitation in the README carries a number and a diagnosis:

- `neutral` F1 is 0.000 for three aspects — **with supports of 3, 8 and 1**, which reframes it from
  a model failure to a data one, and class weighting was tried and moved it not at all.
- Negation is learned per word: `not good` flips (40 negated training examples), `not overpriced`
  does not (3). **The cutoff is about six examples.**
- The `place` over-detection is 18.4% versus 3.2% — and **three fixes were built and all three made
  ambience measurably worse**, so the data was left alone.

That last one is the most useful thing in the section. "I tried three fixes, measured them, and kept
the original because the data said so" is a stronger signal than a clean fix would have been.

---

## 4. One thing this sprint caught

The audit for attribution leaks found `.dockerignore` naming a working-notes file. Listing a filename
in a file that ships with the repo publishes the name — the same mistake made with `.gitignore` in
sprint 1, in a file added eight sprints later.

Replaced with a root-level markdown glob, which excludes the same files without naming any of them.
Worth recording because it is a pattern, not an incident: **anything that excludes something by name
is itself a disclosure.**

---

## 5. The project, finished

| sprint | what |
|---|---|
| 01 | data pipeline, DistilBERT with five aspect heads |
| 02 | MLflow tracking, model registry |
| 03 | evaluation, class weighting, explainability |
| 04 | FastAPI service |
| 05 | 147 tests, layered so 125 need no GPU |
| 06 | per-aspect attention pooling |
| 07 | MAMS supplementary data |
| 08 | React frontend |
| 09 | Docker, CI/CD |
| 10 | documentation |

**Definition of done, checked:** working model → MLflow-tracked with params, P/R/F1, confusion
matrices and a versioned artifact → one explainability method, clearly labelled → FastAPI serving it
→ React page consuming it → Dockerised → CI running tests on push → README with an architecture
diagram. Every piece committed.

---

## 6. What I would say about it in an interview

The score is not the story. Four things happened here that are worth more:

**A component that was silently inert.** Attention pooling shipped doing nothing — scores divided by
`sqrt(d)` collapsed the softmax to exactly uniform, and every accuracy metric looked normal. Only
inspecting the weights found it. The tests now assert on the mechanism rather than the metric.

**An artifact that was never self-contained.** `from_pretrained` downloaded 250 MB of weights that
`load_state_dict` overwrote on the next line, invisible on any machine with a network or a warm
cache. A container with `HF_HUB_OFFLINE=1` found it in one run.

**Four hypotheses that failed.** Attention pooling, filtered MAMS, masking `place`, remapping
`place` — each with sound reasoning, each measured, each kept or discarded on evidence. Two of the
"failures" are still in the codebase as tested-and-rejected options.

**A bug that was my own test input.** Three fixes were built for a wrong prediction on "Cosy little
place" before anyone checked that `cosy` appears zero times in the training corpus and tokenises to
two meaningless fragments. `cozy` gives the correct answer. Validate the probe before blaming the
model.
