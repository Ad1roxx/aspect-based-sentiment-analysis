"""Fine-tune the aspect-sentiment model and save the artifact.

    python ml/src/train.py

Deliberately a hand-written loop rather than HuggingFace's Trainer. Trainer
would supply checkpointing and logging for free, but it hides the optimiser
step, the scheduler and the evaluation behind roughly a hundred constructor
arguments. The loop below is about forty lines and there is nothing in it that
cannot be explained.

MLflow tracking is intentionally absent — that is the next sprint. This script
trains, evaluates and saves; nothing more.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import ASPECTS, IGNORE_INDEX, LABEL_NAMES, Example, load_splits
from model import ENCODER_NAME, MAX_LENGTH, AspectSentimentModel, compute_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "ml" / "models" / "absa-distilbert"


@dataclass
class Config:
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


class AspectDataset(Dataset):
    """Holds raw text; tokenisation happens per batch in the collate function."""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def make_collate_fn(tokenizer: AutoTokenizer):
    """Tokenise a batch, padding only to the longest sentence *in that batch*.

    Padding to a fixed MAX_LENGTH would waste compute on every short review, and
    SemEval sentences are mostly short. Dynamic padding means a batch of
    ten-token sentences runs as ten tokens, not 128.
    """

    def collate(batch: list[Example]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [example.text for example in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([example.labels for example in batch]),
        }

    return collate


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches.

    Without this, two runs with identical hyperparameters produce different
    numbers and there is no way to tell an improvement from noise — which
    matters immediately, because the next sprint compares MLflow runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: AspectSentimentModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return validation loss plus per-aspect macro-F1.

    Macro-F1 averages the F1 of each class equally, so a rare class counts as
    much as a common one. That is the point: 'absent' accounts for roughly 80%
    of labels, so plain accuracy would reward a model that never predicts a
    sentiment at all.
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    true_by_aspect: list[list[int]] = [[] for _ in ASPECTS]
    pred_by_aspect: list[list[int]] = [[] for _ in ASPECTS]

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        total_loss += compute_loss(logits, labels).item()
        batches += 1

        predictions = logits.argmax(dim=-1).cpu()
        labels = labels.cpu()

        for aspect_index in range(len(ASPECTS)):
            aspect_labels = labels[:, aspect_index]
            # Conflict-masked positions were excluded from training, so
            # including them here would score the model on something it was
            # never taught.
            keep = aspect_labels != IGNORE_INDEX
            true_by_aspect[aspect_index].extend(aspect_labels[keep].tolist())
            pred_by_aspect[aspect_index].extend(
                predictions[:, aspect_index][keep].tolist()
            )

    metrics = {"val_loss": total_loss / max(batches, 1)}
    per_aspect = []
    for aspect_index, aspect in enumerate(ASPECTS):
        score = f1_score(
            true_by_aspect[aspect_index],
            pred_by_aspect[aspect_index],
            average="macro",
            labels=list(range(len(LABEL_NAMES))),
            zero_division=0,
        )
        metrics[f"f1_{aspect}"] = score
        per_aspect.append(score)

    metrics["macro_f1"] = float(np.mean(per_aspect))
    return metrics


def save_artifact(
    model: AspectSentimentModel,
    tokenizer: AutoTokenizer,
    config: Config,
    metrics: dict[str, float],
) -> None:
    """Write weights, tokenizer and metadata to ml/models/.

    The tokenizer is saved alongside the weights rather than re-downloaded at
    serving time, so the API container has no network dependency and cannot
    drift onto a different vocabulary than the one trained against.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)

    metadata = {
        "encoder": ENCODER_NAME,
        "aspects": list(ASPECTS),
        "labels": list(LABEL_NAMES),
        "max_length": MAX_LENGTH,
        "hyperparameters": vars(config),
        "validation_metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nsaved to {OUTPUT_DIR}")


def train(config: Config) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_examples, val_examples, _ = load_splits()
    print(f"train: {len(train_examples)}  val: {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    collate = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        AspectDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        AspectDataset(val_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = AspectSentimentModel().to(device)

    # Weight decay is a regulariser that pulls weights toward zero. Applying it
    # to biases and LayerNorm scales hurts — those need the freedom to shift and
    # rescale activations — so they are excluded. This split is standard for
    # transformer fine-tuning.
    no_decay = ("bias", "LayerNorm.weight")
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(marker in n for marker in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(marker in n for marker in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    print(f"steps: {total_steps} ({len(train_loader)} per epoch)\n")
    started = time.time()
    metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = compute_loss(logits, batch["labels"].to(device))
            loss.backward()

            # Fine-tuning occasionally produces a very large gradient that would
            # undo good weights in a single step. Clipping bounds the update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        per_aspect = "  ".join(f"{a}={metrics[f'f1_{a}']:.3f}" for a in ASPECTS)
        print(
            f"epoch {epoch}/{config.epochs}  "
            f"train_loss={epoch_loss / len(train_loader):.4f}  "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        print(f"          {per_aspect}")

    print(f"\ntrained in {time.time() - started:.1f}s")
    save_artifact(model, tokenizer, config, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    train(
        Config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
