"""Loads the model once and answers questions about it.

WHICH LOAD PATH, AND WHY
Sprints 2 and 3 left two ways to load the model, and the API is where that had
to be decided:

  A. from disk      predict.load_model() reads ml/models/absa-distilbert/
  B. from the       mlflow.pyfunc.load_model("models:/absa-distilbert/3")
     registry

**This service uses A.** The reasoning:

* Serving must not depend on MLflow being reachable. Option B makes the tracking
  database a runtime dependency of the API — if sqlite is missing, or the
  artifact store has moved, the service will not start. A prediction endpoint
  should not fail because an experiment-tracking system is down.
* The container gets smaller and starts faster. mlflow pulls in Flask, Alembic,
  SQLAlchemy, GraphQL, Docker and matplotlib, none of which serve a request.
* It keeps one code path with training. predict.load_model() is the same
  function evaluate.py uses, so anything that would break serving breaks
  evaluation first, where it is cheaper to notice.

The registry is not abandoned — it stays the *build-time* source of truth. The
intended pipeline is: choose a version in the registry, fetch it during the
Docker build, bake it into the image. What the running service must never do is
resolve a model over the network at request time.

The cost of choosing A is that the artifact must identify itself, since there is
no URI naming a version. That is why train.py stamps run_id, git_commit and
registry_version into metadata.json, and why /model-info returns them. "Which
model is in production?" is answerable from the running service alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "ml" / "src"

# ml/src uses flat imports (`from data import ...`) rather than being a package,
# so it goes on sys.path rather than being imported as `ml.src.model`. Changing
# that would mean rewriting every import in every training module for the
# benefit of one consumer.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import ASPECTS  # noqa: E402
from explain import explain as explain_aspect  # noqa: E402
from model import MAX_LENGTH  # noqa: E402
from model import predict as run_prediction  # noqa: E402
from predict import load_model  # noqa: E402


class ModelService:
    """Holds the loaded model for the process lifetime.

    Loading happens once at startup, not per request. A DistilBERT checkpoint is
    ~265 MB and takes seconds to read; doing that inside a request handler would
    make every prediction unusably slow and would hold several copies in memory
    under concurrency.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.metadata: dict = {}
        self.device = torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        # CPU on purpose. Serving one short sentence at a time is dominated by
        # overhead, not matrix multiplication, so a GPU buys little while adding
        # a hard deployment constraint. Training is the GPU workload; inference
        # is not.
        self.device = torch.device("cpu")
        self.model, self.tokenizer, self.metadata = load_model(device=self.device)

    def is_truncated(self, text: str) -> bool:
        """Whether the tokenizer had to cut this input to fit MAX_LENGTH.

        Checked with add_special_tokens so the count matches what the model
        actually receives — [CLS] and [SEP] occupy two of the 128 positions.
        """
        token_count = len(self.tokenizer(text, add_special_tokens=True)["input_ids"])
        return token_count > MAX_LENGTH

    def predict(self, text: str, explain: bool = False) -> tuple[list[dict], bool, float]:
        """Predict every aspect, optionally attributing the detected ones.

        Returns (aspects, truncated, latency_ms).

        Explanations are computed ONLY for aspects the model actually detected.
        Each one costs a separate backward pass, so explaining all five would be
        five times the work to produce four results that say "here is why the
        model thinks price was never discussed" — true, and almost never what
        anyone wanted.
        """
        started = time.perf_counter()
        truncated = self.is_truncated(text)

        predictions = run_prediction(self.model, [text], self.tokenizer, self.device)[0]

        aspects: list[dict] = []
        for aspect in ASPECTS:
            result = predictions[aspect]
            mentioned = result["label"] != "absent"

            words = None
            if explain and mentioned:
                attribution = explain_aspect(
                    self.model, text, self.tokenizer, self.device, aspect
                )
                words = [
                    {"word": word, "importance": importance}
                    for word, importance in attribution["words"]
                ]

            aspects.append(
                {
                    "aspect": aspect,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "mentioned": mentioned,
                    "words": words,
                }
            )

        latency_ms = (time.perf_counter() - started) * 1000
        return aspects, truncated, round(latency_ms, 2)

    def info(self) -> dict:
        """Everything the artifact knows about itself, for /model-info."""
        return {
            "encoder": self.metadata.get("encoder", "unknown"),
            "aspects": self.metadata.get("aspects", list(ASPECTS)),
            "labels": self.metadata.get("labels", []),
            "max_length": self.metadata.get("max_length", MAX_LENGTH),
            "registry_version": self.metadata.get("registry_version"),
            "run_id": self.metadata.get("run_id"),
            "git_commit": self.metadata.get("git_commit"),
            "trained_at": self.metadata.get("trained_at"),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "validation_metrics": self.metadata.get("validation_metrics", {}),
        }


# One instance per process, imported by main.py. Not a singleton pattern with
# lazy getters — just a module-level object, because the process only ever needs
# one and pretending otherwise adds indirection without adding capability.
service = ModelService()
