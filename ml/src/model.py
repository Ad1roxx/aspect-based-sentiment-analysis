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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data import ASPECTS, IGNORE_INDEX, LABEL_NAMES, NUM_CLASSES

ENCODER_NAME = "distilbert-base-uncased"
NUM_ASPECTS = len(ASPECTS)
MAX_LENGTH = 128

POOLING_MODES = ("cls", "attention")
DEFAULT_POOLING = "attention"


def attention_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool token representations into one context vector per aspect.

    A module-level function rather than a method so it can be unit-tested with
    synthetic tensors — no 265 MB encoder required. That matters here more than
    usual: this function shipped broken once (see below) and the bug was
    invisible in every accuracy metric.

    ``hidden_states`` (batch, tokens, dim), ``attention_mask`` (batch, tokens),
    ``queries`` (aspects, dim). Returns (context, weights).

    Scores are NOT divided by sqrt(dim). The 1/sqrt(d) factor in standard
    transformer attention stops softmax saturating when queries and keys both
    have unit-variance entries. Here the keys are raw encoder hidden states and
    the queries start at std=0.02, so scores are already small — dividing by 27.7
    as well collapsed the softmax to exactly uniform (max/min weight 1.014, which
    is mean pooling wearing a costume) and shrank the gradient by the same factor
    so the queries never learned out of it.
    """
    scores = torch.einsum("btd,ad->bat", hidden_states, queries)

    # Padding must score -inf so softmax gives it exactly zero weight. Without
    # this, a short sentence in a long batch pools over padding and its
    # prediction depends on what else happened to be in the batch.
    scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    context = torch.einsum("bat,btd->bad", weights, hidden_states)
    return context, weights


class AspectSentimentModel(nn.Module):
    """Shared DistilBERT encoder + one 4-way classifier per aspect.

    ``pooling`` selects how a sentence becomes the vector each aspect head reads:

    ``cls``        every head reads the same [CLS] vector. Simple, and the cause of
                   a measured failure — see the module docstring.
    ``attention``  each aspect has a learned query and attends over the token
                   representations itself, so each head reads its own evidence.
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        num_aspects: int = NUM_ASPECTS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        pooling: str = DEFAULT_POOLING,
        encoder_config=None,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLING_MODES}")

        # from_config when the caller has one, from_pretrained otherwise.
        #
        # This matters more than it looks. from_pretrained DOWNLOADS the base
        # encoder from the HuggingFace Hub — and load_state_dict then overwrites
        # every one of those weights with ours. So at serving time it fetched
        # 250 MB over the network purely to obtain an architecture we already
        # know, and the artifact was not self-contained at all. That surfaced
        # only when the container ran with HF_HUB_OFFLINE=1 and failed to start.
        #
        # Training still uses from_pretrained: fine-tuning genuinely needs the
        # pretrained weights. Loading a finished artifact does not.
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.pooling = pooling

        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            # One learned query vector per aspect. This is the entire mechanism:
            # the query is what "asking about service" means, and it is learned
            # from the same supervision as everything else. 5 x 768 = 3,840
            # parameters, which is 0.006% of the model.
            self.aspect_queries = nn.Parameter(torch.empty(num_aspects, hidden_size))
            # std=0.02 gives an initial score spread of roughly 1 nat across
            # tokens, which is enough for softmax to distinguish them without
            # starting so peaked that most tokens get no gradient.
            nn.init.normal_(self.aspect_queries, std=0.02)

        # A per-aspect classifier held as one batched tensor rather than five
        # Linear layers. Identical parameterisation to the previous
        # Linear(hidden, num_aspects * num_classes) — every aspect still has its
        # own independent weights and no aspect's logits touch another's — but
        # indexed by aspect so it can consume a per-aspect context vector.
        self.classifier_weight = nn.Parameter(
            torch.empty(num_aspects, hidden_size, num_classes)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(num_aspects, num_classes))
        # Match nn.Linear's default init so the two pooling modes start from
        # comparable distributions and the comparison isolates the pooling.
        bound = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.classifier_weight, -bound, bound)
        nn.init.uniform_(self.classifier_bias, -bound, bound)

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Turn token representations into one context vector per aspect.

        Returns (context, attention_weights). context is (batch, aspect, hidden);
        attention_weights is (batch, aspect, tokens) for attention pooling and
        None for [CLS] pooling — because with [CLS] there is nothing per-aspect
        to report, which is precisely the limitation this mode has.
        """
        if self.pooling == "cls":
            # DistilBERT ships no pooler layer, unlike BERT, so pooling is our
            # choice. Position 0 is the [CLS] token, the conventional sentence
            # summary. Broadcast to every aspect: all five heads read the SAME
            # vector, which is the behaviour under test.
            sentence_vector = hidden_states[:, 0]
            context = sentence_vector.unsqueeze(1).expand(-1, self.num_aspects, -1)
            return context, None

        return attention_pool(hidden_states, attention_mask, self.aspect_queries)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits shaped (batch, num_aspects, num_classes).

        With ``return_attention``, also returns the per-aspect attention weights,
        which is what makes attention a legitimate explanation for THIS
        architecture — each aspect genuinely has its own distribution over tokens.

        Note that ``token_type_ids`` is deliberately not accepted. DistilBERT
        has no segment embeddings, so its forward signature ignores them — but
        the tokenizer still emits them, and passing the whole tokenizer output
        through would let them disappear silently into **kwargs. Naming the two
        tensors we actually use makes that explicit.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        context, weights = self.pool(outputs.last_hidden_state, attention_mask)

        # einsum over the aspect axis: each aspect's context vector meets that
        # aspect's own weight matrix. No aspect's logits touch another's.
        logits = torch.einsum(
            "bad,adc->bac", self.dropout(context), self.classifier_weight
        )
        logits = logits + self.classifier_bias

        if return_attention:
            return logits, weights
        return logits


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
