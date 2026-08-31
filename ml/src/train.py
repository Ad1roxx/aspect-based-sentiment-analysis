"""Fine-tune the aspect-sentiment model and save the artifact.

    python ml/src/train.py

Deliberately a hand-written loop rather than HuggingFace's Trainer. Trainer
would supply checkpointing and logging for free, but it hides the optimiser
step, the scheduler and the evaluation behind roughly a hundred constructor
arguments. The loop below is about forty lines and there is nothing in it that
cannot be explained.

Every run is recorded to MLflow (see tracking.py): params, per-epoch metrics,
per-class F1, confusion matrices, and the model itself as a registered version.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import tracking
from mams import MODES as EXTRA_DATA_MODES
from mams import load_mams
from data import (
    ABSENT,
    ASPECTS,
    IGNORE_INDEX,
    LABEL_NAMES,
    Example,
    class_counts,
    load_splits,
)
from model import (
    DEFAULT_POOLING,
    ENCODER_NAME,
    MAX_LENGTH,
    POOLING_MODES,
    AspectSentimentModel,
    compute_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "ml" / "models" / "absa-distilbert"

WEIGHT_SCHEMES = ("none", "sqrt-inverse", "inverse")


def class_weight_tensor(
    examples: list[Example],
    scheme: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Build per-class loss weights from the training distribution.

    ``inverse``       w_c = N / (C * n_c) — the textbook balanced weighting. It
                      equalises each class's total contribution to the loss, but
                      with a 22:1 imbalance it hands 'neutral' a weight of ~7.3,
                      which can push the model into predicting rare classes
                      constantly and destroying precision.
    ``sqrt-inverse``  the square root of the above, renormalised to mean 1. A
                      standard damped variant: it corrects in the same direction
                      with roughly a quarter of the force.

    Weights come from the TRAINING split only. Computing them over train+val
    would leak the validation distribution into a training decision — a small
    leak, but the kind that makes a validation score optimistic for no reason.
    """
    if scheme == "none":
        return None
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {WEIGHT_SCHEMES}")

    counts = np.array(class_counts(examples), dtype=np.float64)
    if (counts == 0).any():
        raise ValueError(f"a class is unrepresented in training: {counts.tolist()}")

    weights = counts.sum() / (len(counts) * counts)
    if scheme == "sqrt-inverse":
        weights = np.sqrt(weights)
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


@dataclass
class Config:
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    # One of WEIGHT_SCHEMES. Defaults to "none" so the baseline stays the
    # baseline; changing a default silently would make every prior run
    # incomparable to every future one.
    class_weights: str = "none"
    # 'cls' or 'attention' — see model.AspectSentimentModel.
    pooling: str = DEFAULT_POOLING
    # Applies only to the attention queries; ignored for 'cls' pooling.
    query_learning_rate: float = 1e-3
    # Supplementary MAMS-ACSA data: 'none', 'filtered' or 'full'. See ml/src/mams.py.
    # Only ever added to TRAIN — validation and test stay pure SemEval-2014, or the
    # numbers would no longer be comparable with every earlier run.
    extra_data: str = "none"
    # Off by default on purpose. The test split is the estimate of how the model
    # does on data it has never influenced; evaluating against it on every run
    # while tuning turns it into a second validation set, and the number stops
    # meaning anything. Turn it on deliberately, at the end.
    eval_test: bool = False


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
) -> tuple[dict[str, float], list[list[int]], list[list[int]]]:
    """Return validation loss, per-aspect macro-F1, and the raw label/prediction
    lists that confusion matrices and per-class reports are built from.

    Macro-F1 averages the F1 of each class equally, so a rare class counts as
    much as a common one. That is the point: 'absent' accounts for roughly 80%
    of labels, so plain accuracy would reward a model that never predicts a
    sentiment at all.

    The raw lists are returned rather than recomputed later because the model is
    already in eval mode with the data already batched — running the whole split
    a second time just to draw a heatmap would be wasteful, and worse, would risk
    scoring a model that had since been mutated.
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    true_by_aspect: list[list[int]] = [[] for _ in ASPECTS]
    pred_by_aspect: list[list[int]] = [[] for _ in ASPECTS]
    mention_hits = {"single": 0, "multi": 0}
    mention_totals = {"single": 0, "multi": 0}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        # Deliberately UNWEIGHTED, even when training is weighted. Validation
        # loss is used to compare runs against each other, and a loss computed
        # under different weights is a different quantity — run A scoring lower
        # than run B would tell you nothing. Macro-F1 is the selection metric;
        # this stays a fixed yardstick.
        total_loss += compute_loss(logits, labels).item()
        batches += 1

        predictions = logits.argmax(dim=-1).cpu()
        labels = labels.cpu()

        # Mention-detection recall, split by how many aspects the sentence
        # actually discusses. This is the metric the architecture change targets:
        # of aspects genuinely discussed, how often does the model avoid calling
        # them 'absent'? Macro-F1 averages this failure away, because a lost
        # aspect looks like a correct 'absent' prediction for four other aspects.
        real = (labels != ABSENT) & (labels != IGNORE_INDEX)
        found = real & (predictions != ABSENT)
        mentioned_per_row = real.sum(dim=1)
        for row in range(labels.size(0)):
            count = int(mentioned_per_row[row])
            if count == 0:
                continue
            bucket = "single" if count == 1 else "multi"
            mention_hits[bucket] += int(found[row].sum())
            mention_totals[bucket] += count

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

    for bucket in ("single", "multi"):
        total = mention_totals[bucket]
        metrics[f"mention_recall_{bucket}"] = (
            mention_hits[bucket] / total if total else 0.0
        )
    # The gap is the headline number for this experiment: how much worse the
    # model gets when a sentence discusses more than one thing.
    metrics["mention_recall_gap"] = (
        metrics["mention_recall_single"] - metrics["mention_recall_multi"]
    )

    return metrics, true_by_aspect, pred_by_aspect


def git_commit() -> str | None:
    """The commit the artifact was trained from, or None outside a repo.

    Params record the hyperparameters; the commit records the code. Two runs with
    identical params can still differ if the loss function changed between them,
    and this is the only field that catches that.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def stamp_registry_version(model_dir: Path, version: str) -> None:
    """Record the assigned registry version into the written metadata.json."""
    path = model_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["registry_version"] = version
    path.write_text(json.dumps(metadata, indent=2))


def save_artifact(
    model: AspectSentimentModel,
    tokenizer: AutoTokenizer,
    config: Config,
    metrics: dict[str, float],
    run_id: str,
) -> None:
    """Write weights, tokenizer and metadata to ml/models/.

    The tokenizer is saved alongside the weights rather than re-downloaded at
    serving time, so the API container has no network dependency and cannot
    drift onto a different vocabulary than the one trained against.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)
    # The encoder's architecture description. Without it, loading the artifact
    # has to call from_pretrained, which reaches out to the HuggingFace Hub for
    # weights that load_state_dict then immediately replaces. Saving ~1 KB of
    # JSON here is what makes the artifact loadable with no network at all.
    model.encoder.config.save_pretrained(OUTPUT_DIR)

    metadata = {
        "encoder": ENCODER_NAME,
        "aspects": list(ASPECTS),
        "labels": list(LABEL_NAMES),
        "max_length": MAX_LENGTH,
        # Load-bearing: the artifact cannot be rebuilt without knowing which
        # pooling produced it, because the two modes have different parameters.
        "pooling": config.pooling,
        "hyperparameters": vars(config),
        "validation_metrics": {k: round(v, 4) for k, v in metrics.items()},
        # Provenance. The API serves this directory, not the MLflow registry, so
        # the artifact must be able to say what it is on its own — otherwise
        # "which model is in production?" is unanswerable from the running
        # service. registry_version stays None unless the run was promoted.
        "run_id": run_id,
        "git_commit": git_commit(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry_version": None,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nsaved to {OUTPUT_DIR}")


def train(
    config: Config,
    run_name: str | None = None,
    register: bool = False,
) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_examples, val_examples, test_examples = load_splits()

    if config.extra_data != "none":
        extra = load_mams(config.extra_data, verbose=True)
        # Appended to TRAIN only. Validation and test remain pure SemEval-2014 so
        # every number stays comparable with the runs from sprints 1-6.
        train_examples = train_examples + extra

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

    model = AspectSentimentModel(pooling=config.pooling).to(device)
    weights = class_weight_tensor(train_examples, config.class_weights, device)
    if weights is not None:
        formatted = ", ".join(
            f"{name}={value:.2f}" for name, value in zip(LABEL_NAMES, weights.tolist())
        )
        print(f"class weights ({config.class_weights}): {formatted}")

    # Weight decay is a regulariser that pulls weights toward zero. Applying it
    # to biases and LayerNorm scales hurts — those need the freedom to shift and
    # rescale activations — so they are excluded. This split is standard for
    # transformer fine-tuning.
    no_decay = ("bias", "LayerNorm.weight")

    # aspect_queries gets its own, much higher learning rate. The encoder is
    # pretrained and only needs nudging, which is what 2e-5 is for; the aspect
    # queries are randomly initialised and have to travel a long way in 324 steps.
    # At 2e-5 they do not move at all — measured: their norm after four epochs was
    # 0.55, identical to initialisation, and the attention stayed near-uniform.
    # Separate rates for pretrained bodies and fresh heads is standard practice
    # and is the difference between this mechanism working and being decoration.
    fresh = ("aspect_queries",)

    def is_fresh(name: str) -> bool:
        return any(marker in name for marker in fresh)

    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not is_fresh(n) and not any(m in n for m in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not is_fresh(n) and any(m in n for m in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": config.learning_rate,
        },
    ]
    fresh_params = [p for n, p in model.named_parameters() if is_fresh(n)]
    if fresh_params:
        grouped_parameters.append(
            {
                "params": fresh_params,
                "weight_decay": 0.0,
                "lr": config.query_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    print(f"steps: {total_steps} ({len(train_loader)} per epoch)")
    print(f"tracking: {tracking.configure()}\n")

    with mlflow.start_run(run_name=run_name) as run:
        # Params are the inputs to the run; tags are facts about the context it
        # ran in. MLflow lets you filter and sort on both, which is what makes
        # "show me every run with lr=3e-5 on the 4060" a query instead of a
        # memory exercise.
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "encoder": ENCODER_NAME,
                "max_length": MAX_LENGTH,
                "total_steps": total_steps,
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )
        mlflow.set_tags(
            tracking.environment_tags(len(train_examples), len(val_examples))
        )
        print(f"run_id: {run.info.run_id}\n")

        started = time.time()
        metrics: dict[str, float] = {}
        true_by_aspect: list[list[int]] = []
        pred_by_aspect: list[list[int]] = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = compute_loss(logits, batch["labels"].to(device), weights)
                loss.backward()

                # Fine-tuning occasionally produces a very large gradient that
                # would undo good weights in a single step. Clipping bounds it.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            metrics, true_by_aspect, pred_by_aspect = evaluate(model, val_loader, device)

            # step=epoch turns these into a curve rather than a final number, so
            # overfitting shows up as val_loss turning back up while train_loss
            # keeps falling — visible in the MLflow chart, invisible in a summary.
            mlflow.log_metrics({"train_loss": train_loss, **metrics}, step=epoch)

            per_aspect = "  ".join(f"{a}={metrics[f'f1_{a}']:.3f}" for a in ASPECTS)
            print(
                f"epoch {epoch}/{config.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={metrics['val_loss']:.4f}  "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            print(f"          {per_aspect}")
            print(
                f"          mention recall: single={metrics['mention_recall_single']:.3f}"
                f"  multi={metrics['mention_recall_multi']:.3f}"
                f"  gap={metrics['mention_recall_gap']:.3f}"
            )

        elapsed = time.time() - started
        mlflow.log_metric("training_seconds", elapsed)
        print(f"\ntrained in {elapsed:.1f}s")

        tracking.log_evaluation(true_by_aspect, pred_by_aspect, "val")

        if config.eval_test:
            test_loader = DataLoader(
                AspectDataset(test_examples),
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate,
            )
            test_metrics, test_true, test_pred = evaluate(model, test_loader, device)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            tracking.log_evaluation(test_true, test_pred, "test")
            print(f"test macro_f1={test_metrics['macro_f1']:.4f}")

        # Save to disk first: log_model reads the artifact directory back off
        # disk to package it, so the ordering is a real dependency, not a style.
        save_artifact(model, tokenizer, config, metrics, run.info.run_id)
        version = tracking.log_model(OUTPUT_DIR, register=register)

        # Stamped after registration: the version number does not exist until
        # the registry assigns it.
        if version is not None:
            stamp_registry_version(OUTPUT_DIR, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="also score the held-out test split (use sparingly — see Config)",
    )
    parser.add_argument(
        "--class-weights",
        choices=WEIGHT_SCHEMES,
        default=defaults.class_weights,
        help="per-class loss weighting to counteract the ~77%% 'absent' majority",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="label for this MLflow run (defaults to an MLflow-generated name)",
    )
    parser.add_argument(
        "--pooling",
        choices=POOLING_MODES,
        default=defaults.pooling,
        help="how a sentence becomes the vector each aspect head reads",
    )
    parser.add_argument(
        "--extra-data",
        choices=EXTRA_DATA_MODES,
        default=defaults.extra_data,
        help="add MAMS-ACSA multi-aspect sentences to training",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="promote this run's model to a new Model Registry version",
    )
    args = parser.parse_args()

    train(
        Config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            eval_test=args.eval_test,
            class_weights=args.class_weights,
            pooling=args.pooling,
            extra_data=args.extra_data,
        ),
        run_name=args.run_name,
        register=args.register,
    )


if __name__ == "__main__":
    main()
