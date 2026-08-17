"""Score a saved model artifact without retraining it.

    python ml/src/evaluate.py                  # test split (the default)
    python ml/src/evaluate.py --split val
    python ml/src/evaluate.py --save-figure

Training already evaluates, so why does this exist? Because training evaluates
*the model it just built in memory*, and that is a different thing from the
artifact sitting on disk. This script scores what was actually saved — the same
files the API will load — which is the only way to catch a broken save step, a
tokenizer/weights mismatch, or a model that was silently overwritten by a later
run. It is also how you re-score a model months later without the GPU, the
training data pipeline, or 50 seconds of fine-tuning.

Loads from ``ml/models/absa-distilbert/`` via predict.load_model — the same disk
path predict.py uses. It does not read from the MLflow registry; see the
"two load paths" note in explanations/sprint-02.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import ASPECTS, load_splits
from evaluation import classification_report_text, confusion_matrix_figure
from predict import load_model
from train import AspectDataset, make_collate_fn
from train import evaluate as score_model

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="which split to score (default: test)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help=f"write the confusion-matrix grid to {REPORT_DIR.relative_to(REPO_ROOT)}",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, metadata = load_model(device=device)

    splits = dict(zip(("train", "val", "test"), load_splits()))
    examples = splits[args.split]

    loader = DataLoader(
        AspectDataset(examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )
    metrics, true_by_aspect, pred_by_aspect = score_model(model, loader, device)

    print(f"artifact  : {metadata['encoder']}")
    print(f"trained on: {metadata['hyperparameters']}")
    print(f"split     : {args.split}  ({len(examples)} sentences)\n")

    print(classification_report_text(true_by_aspect, pred_by_aspect))

    print(f"{'aspect':<12}{'macro-F1':>10}")
    print("-" * 22)
    for aspect in ASPECTS:
        print(f"{aspect:<12}{metrics[f'f1_{aspect}']:>10.4f}")
    print("-" * 22)
    print(f"{'OVERALL':<12}{metrics['macro_f1']:>10.4f}")
    print(f"{'loss':<12}{metrics['val_loss']:>10.4f}")

    if args.save_figure:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        destination = REPORT_DIR / f"confusion_{args.split}.png"
        figure = confusion_matrix_figure(true_by_aspect, pred_by_aspect, args.split)
        figure.savefig(destination, dpi=120)
        print(f"\nfigure -> {destination}")


if __name__ == "__main__":
    main()
