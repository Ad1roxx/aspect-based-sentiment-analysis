"""DistilBERT with five aspect heads.

One shared encoder reads the sentence once; five small classifiers then each
answer a 4-way question about it (absent / negative / neutral / positive).

Sharing the encoder is the whole point. Fine-tuning five separate DistilBERTs
would cost five times the memory and five times the inference latency, and each
would have to relearn English from a fifth of the supervision. Whether a
sentence is negative about service and whether it praises the food are largely
the same reading task, so the representation transfers.

Run directly for a shape and parameter-count check::

    python ml/src/model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from data import ASPECTS, IGNORE_INDEX, LABEL_NAMES, NUM_CLASSES

ENCODER_NAME = "distilbert-base-uncased"
NUM_ASPECTS = len(ASPECTS)
MAX_LENGTH = 128


class AspectSentimentModel(nn.Module):
    """Shared DistilBERT encoder + one 4-way classifier per aspect."""

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        num_aspects: int = NUM_ASPECTS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.num_aspects = num_aspects
        self.num_classes = num_classes

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # One Linear producing num_aspects * num_classes logits, reshaped to
        # (batch, aspect, class). This is mathematically identical to five
        # separate Linear(hidden, 4) layers — every output unit already has its
        # own independent row of weights, and no aspect's logits influence
        # another's. One matrix multiply is simply faster than five.
        self.classifier = nn.Linear(hidden_size, num_aspects * num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped (batch, num_aspects, num_classes).

        Note that ``token_type_ids`` is deliberately not accepted. DistilBERT
        has no segment embeddings, so its forward signature ignores them — but
        the tokenizer still emits them, and passing the whole tokenizer output
        through would let them disappear silently into **kwargs. Naming the two
        tensors we actually use makes that explicit.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT ships no pooler layer, unlike BERT, so pooling is our
        # choice. Position 0 is the [CLS] token, whose representation attends
        # over the entire sequence and is the conventional sentence summary.
        sentence_vector = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(sentence_vector))
        return logits.view(-1, self.num_aspects, self.num_classes)


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean cross-entropy over every (sentence, aspect) pair.

    Flattening (batch, aspect, class) to (batch * aspect, class) treats each
    aspect prediction as its own training example, which is exactly the intent:
    five independent classifications that happen to share an encoder.

    ``ignore_index`` drops the positions labelled IGNORE_INDEX — the "conflict"
    annotations — from both the loss and its gradient, so those aspects
    contribute nothing rather than contributing something wrong.

    ``class_weights`` multiplies each class's contribution to the loss. With
    'absent' at 77% of labels and 'neutral' at 3%, the unweighted gradient is
    dominated by a class the model finds easy, and predicting 'neutral' is never
    worth the risk. Upweighting rare classes makes missing them expensive.
    Whether that actually improves macro-F1 is an empirical question, not an
    assumption — see explanations/sprint-03.md.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def predict(
    model: AspectSentimentModel,
    texts: list[str],
    tokenizer: AutoTokenizer,
    device: torch.device | str = "cpu",
) -> list[dict[str, dict[str, float | str]]]:
    """Predict every aspect for a batch of raw sentences.

    Returns, per sentence, a mapping of aspect -> {label, confidence}. The
    confidence is the softmax probability of the chosen class, which is what the
    UI displays. Worth being precise about what that number means: it is the
    model's relative preference among four options, not a calibrated probability
    of being correct. Neural classifiers are typically overconfident.
    """
    model.eval()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    logits = model(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
    )
    probabilities = F.softmax(logits, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)

    results = []
    for row in range(len(texts)):
        results.append(
            {
                aspect: {
                    "label": LABEL_NAMES[predictions[row, i].item()],
                    "confidence": round(confidences[row, i].item(), 4),
                }
                for i, aspect in enumerate(ASPECTS)
            }
        )
    return results


def main() -> None:
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AspectSentimentModel()

    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())
    print(f"encoder      : {ENCODER_NAME}")
    print(f"total params : {total:,}")
    print(f"head params  : {head:,}  ({head / total:.2%} of the model)")

    texts = [
        "Great food but the service was painfully slow.",
        "I live a block away and go there often.",
    ]
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    print(f"\nlogits shape : {tuple(logits.shape)}  (batch, aspects, classes)")

    # Loss sanity check: a batch where one aspect is masked out should train on
    # nine of the ten (sentence, aspect) pairs, not ten.
    labels = torch.tensor(
        [
            [3, 1, 0, 0, 0],             # positive food, negative service
            [0, 0, 0, 0, IGNORE_INDEX],  # misc masked as conflict
        ]
    )
    print(f"loss         : {compute_loss(logits, labels).item():.4f}")

    print("\nUntrained predictions (should be near-random):")
    for text, result in zip(texts, predict(model, texts, tokenizer)):
        print(f"  {text}")
        for aspect, out in result.items():
            print(f"    {aspect:<10} {out['label']:<9} {out['confidence']:.3f}")


if __name__ == "__main__":
    main()
