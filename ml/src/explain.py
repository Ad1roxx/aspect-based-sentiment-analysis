"""Per-aspect token attribution: which words drove each aspect's prediction.

    python ml/src/explain.py
    python ml/src/explain.py --text "Great food but the waiter was rude."
    python ml/src/explain.py --text "..." --aspect service

WHY NOT ATTENTION
The obvious choice for a transformer is to show attention weights, and it is
the wrong one *for this architecture*. The model pools a single [CLS] vector and
feeds it to all five aspect heads. Last-layer attention from [CLS] is therefore
one distribution shared by every aspect — it is identical whether you ask about
food or about service. Rendering it on five aspect cards would show five
identical highlights, which is not merely uninformative, it is misleading: it
looks like a per-aspect explanation and is not one.

Attention is also contested as an explanation in general (Jain & Wallace, 2019,
"Attention is not Explanation"): attention weights can be substantially altered
without changing the prediction, so they do not reliably indicate what the model
depended on.

WHAT THIS DOES INSTEAD
Gradient x input attribution. For a chosen aspect and its predicted class, take
the gradient of that logit with respect to the input word embeddings, and dot it
with the embeddings themselves:

    attribution(token) = sum_d ( d logit / d embedding[token, d] ) * embedding[token, d]

The gradient says "how much would this logit change if this embedding moved",
and dotting with the actual embedding converts that sensitivity into a
contribution from the value actually present. Because the gradient is taken from
*one aspect's* logit, the result genuinely differs per aspect. Cost is one
forward and one backward pass.

MAGNITUDE ONLY — AND WHY
The raw attribution is signed, and the sign is NOT reported, because measurement
showed it cannot be trusted. On "…a bit overpriced for what you get", the word
"overpriced" received the largest attribution by magnitude but with a *negative*
sign on a `price -> negative` prediction, implying it argued against the class it
obviously drove.

The reason is structural: gradient x input approximates f(x) - f(0), a Taylor
expansion around an all-zero embedding. The zero vector is not a token, so
"contribution relative to the zero embedding" has no linguistic meaning and its
sign is an artefact. What survives is the magnitude — how sharply this logit
responds to this position.

That magnitude was validated causally by occlusion, replacing the top-attributed
word with [MASK] and re-predicting:

    "…a bit overpriced for what you get"  price: negative 69.8%
    "…a bit [MASK] for what you get"      price: absent   75.6%   (flipped)

    "The pasta was incredible but…"       food:  positive 73.0%
    "The pasta was [MASK] but…"           food:  negative 29.4%   (flipped)

Removing the highest-attributed word changes the prediction in both cases, so
the ranking identifies words the model genuinely depends on. Occlusion is the
more trustworthy method — it is causal rather than a linear approximation — but
it costs one forward pass per word, whereas this costs one per request. Using
the cheap method and validating it against the expensive one is the trade.

HONEST LIMITATIONS — read these before quoting the output
* This is a *local, first-order* approximation. It describes the model's
  sensitivity around this one input, not a global rule it follows.
* Importance is unsigned. It says a word mattered, not which way it pushed.
* Gradient x input is noisier than Integrated Gradients, which integrates along a
  path from a real baseline and satisfies a completeness axiom. IG costs 20-50
  forward passes; this costs one, which is what makes it viable inside an API
  request.
* An attribution is not a justification. It shows what the model used, which is
  as useful for finding spurious correlations as for building confidence.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from data import ASPECTS, LABEL_NAMES
from model import MAX_LENGTH, AspectSentimentModel
from predict import load_model

DEMO_TEXTS = [
    "The pasta was incredible but the waiter ignored us for twenty minutes.",
    "Cosy little place, though it's a bit overpriced for what you get.",
]

SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")


def merge_subwords(
    tokens: list[str],
    scores: list[float],
) -> tuple[list[str], list[float]]:
    """Recombine WordPiece fragments into whole words, summing their scores.

    DistilBERT splits "overpriced" into "over", "##pric", "##ed". Showing those
    three fragments separately is unreadable, and worse, splits one word's
    contribution across three bars so each looks less important than the word
    actually was. Attributions are additive in the embedding dimension, so
    summing the fragments is the right aggregation rather than averaging.

    Special tokens are dropped: [CLS] usually carries large attribution simply
    because it is the pooled position, which says nothing about the input text.
    """
    words: list[str] = []
    merged: list[float] = []

    for token, score in zip(tokens, scores):
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("##") and words:
            words[-1] += token[2:]
            merged[-1] += score
        else:
            words.append(token)
            merged.append(score)

    return words, merged


def explain(
    model: AspectSentimentModel,
    text: str,
    tokenizer,
    device: torch.device | str,
    aspect: str,
) -> dict:
    """Attribute one aspect's prediction to the input words.

    Returns the predicted label, its confidence, and a (word, importance) list
    where importance is unsigned and normalised so the strongest word scores
    1.0. See the module docstring for why the sign is discarded.
    """
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}; choose from {list(ASPECTS)}")
    aspect_index = ASPECTS.index(aspect)

    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # A forward hook is used rather than passing inputs_embeds, so the model's
    # own forward signature stays unchanged — explainability is an observer here,
    # not something the architecture has to accommodate.
    captured: dict[str, torch.Tensor] = {}

    def capture(module, inputs, output):
        # The embedding output is a non-leaf tensor, so its .grad is discarded
        # after backward unless retained explicitly.
        output.retain_grad()
        captured["embeddings"] = output

    handle = model.encoder.embeddings.word_embeddings.register_forward_hook(capture)
    try:
        # No torch.no_grad() here: the whole method depends on the graph that
        # inference normally throws away.
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=-1)

        predicted = int(logits[0, aspect_index].argmax())
        # detach() before float(): probabilities is part of the live autograd
        # graph, and converting a grad-tracking tensor to a Python scalar warns.
        confidence = float(probabilities[0, aspect_index, predicted].detach())

        model.zero_grad(set_to_none=True)
        # Backward from this single logit, not from a loss. This is what makes
        # the attribution aspect-specific: a different aspect_index produces a
        # different gradient and therefore a different explanation.
        logits[0, aspect_index, predicted].backward()
    finally:
        handle.remove()

    embeddings = captured["embeddings"].detach()[0]
    gradients = captured["embeddings"].grad[0]
    attributions = (gradients * embeddings).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    words, scores = merge_subwords(tokens, attributions.tolist())

    # Subwords are summed as signed values (that is the correct aggregation),
    # and only then reduced to magnitude. Taking absolute values first would let
    # two fragments of one word cancel or inflate each other arbitrarily.
    scores = [abs(score) for score in scores]

    # Normalise by the largest so the strongest word is 1.0 and the scale is
    # comparable across sentences and aspects. Dividing by the sum would instead
    # make importance depend on sentence length.
    largest = max(scores, default=0.0)
    if largest > 0:
        scores = [score / largest for score in scores]

    return {
        "aspect": aspect,
        "label": LABEL_NAMES[predicted],
        "confidence": round(confidence, 4),
        "words": list(zip(words, [round(s, 4) for s in scores])),
    }


def render(result: dict, width: int = 30, top_k: int = 8) -> None:
    """Print a bar chart of word importances, strongest first."""
    print(f"  {result['aspect']} -> {result['label']} ({result['confidence']:.1%})")

    ranked = sorted(result["words"], key=lambda pair: -pair[1])
    for word, score in ranked[:top_k]:
        bar = "#" * max(int(score * width), 1)
        print(f"    {word:<16}{score:>6.3f}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="review to explain (repeatable)")
    parser.add_argument(
        "--aspect",
        choices=ASPECTS,
        help="explain only this aspect (default: every aspect the model detects)",
    )
    args = parser.parse_args()

    texts = args.text or DEMO_TEXTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model(device=device)

    for text in texts:
        print(f"\n{text}")
        print("-" * len(text))

        if args.aspect:
            render(explain(model, text, tokenizer, device, args.aspect))
            continue

        # Default: explain only the aspects the model actually detected.
        # Attributing an 'absent' prediction is legal but rarely interesting —
        # it explains why the model thinks a topic was not discussed.
        detected = False
        for aspect in ASPECTS:
            result = explain(model, text, tokenizer, device, aspect)
            if result["label"] != "absent":
                render(result)
                detected = True
        if not detected:
            print("  (no aspects detected)\n")


if __name__ == "__main__":
    main()
