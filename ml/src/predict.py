"""Load the saved artifact and predict aspect sentiments for a review.

    python ml/src/predict.py
    python ml/src/predict.py --text "The pasta was great but we waited an hour."

This exists to prove the round trip: a model that trains but cannot be reloaded
from disk is useless to the API. Everything here is what the FastAPI service
will do at startup, minus the web layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import AspectSentimentModel, predict

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "ml" / "models" / "absa-distilbert"

DEMO_TEXTS = [
    "The pasta was incredible but the waiter ignored us for twenty minutes.",
    "Cosy little place, though it's a bit overpriced for what you get.",
    "I walked past it on my way to work.",
]


def load_model(
    model_dir: Path = MODEL_DIR,
    device: torch.device | str = "cpu",
) -> tuple[AspectSentimentModel, AutoTokenizer, dict]:
    """Rebuild the model from disk.

    ``weights_only=True`` tells torch.load to refuse anything but plain tensors.
    A checkpoint is a pickle, and unpickling executes arbitrary code, so a model
    file from an untrusted source could run anything. Ours is self-produced, but
    the serving path should not depend on that being remembered.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{model_dir} not found. Train first: python ml/src/train.py"
        )

    metadata = json.loads((model_dir / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AspectSentimentModel(encoder_name=metadata["encoder"])
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    return model, tokenizer, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="review to analyse (repeatable)")
    args = parser.parse_args()

    texts = args.text or DEMO_TEXTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, metadata = load_model(device=device)
    print(f"loaded {metadata['encoder']} from {MODEL_DIR.name}")
    print(f"validation macro-F1: {metadata['validation_metrics']['macro_f1']}\n")

    for text, result in zip(texts, predict(model, texts, tokenizer, device)):
        print(text)
        for aspect, out in result.items():
            marker = " " if out["label"] == "absent" else "*"
            print(f"  {marker} {aspect:<10} {out['label']:<9} {out['confidence']:>6.1%}")
        print()


if __name__ == "__main__":
    main()
