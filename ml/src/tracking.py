"""MLflow experiment tracking: what was run, what it scored, what came out.

Sprint 1 printed metrics to the terminal, which is fine for exactly one run and
useless for two. The moment we start tuning — a different learning rate, class
weights for the neutral collapse — "which run was that, and what did I change?"
becomes unanswerable from scrollback. This module is the answer to that.

Three things get recorded per run:

  params   the full Config plus the environment it ran in (device, versions,
           dataset sizes). Enough to re-run it.
  metrics  loss and macro-F1 per epoch, then per-aspect and per-class F1 at the
           end. Per-class is the point — the headline macro-F1 hides that
           'neutral' scores 0.000 on four of five aspects.
  artifacts confusion-matrix grid, a full precision/recall/F1 table, and the
           model itself, registered so it gets a real version number.

BACKEND STORE
The tracking URI is sqlite, not the classic ./mlruns directory. That is not a
preference: as of MLflow 3.x the filesystem store is in maintenance mode and
raises on use unless MLFLOW_ALLOW_FILE_STORE is set, and it never supported the
Model Registry at all. sqlite is a one-line change that keeps both working.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import mlflow
import torch

# Imported before pyplot on purpose. evaluation.py selects matplotlib's
# non-interactive "Agg" backend, and a backend must be chosen before pyplot is
# first imported anywhere in the process. Importing evaluation first guarantees
# that ordering no matter who imports tracking.
from evaluation import (
    classification_report_text,
    confusion_matrix_figure,
    per_class_metrics,
)
from serving import ABSAModel

import matplotlib.pyplot as plt  # noqa: E402  — see the ordering note above

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent

# sqlite URIs need forward slashes even on Windows, hence as_posix().
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'ml' / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "absa-restaurants"
REGISTERED_MODEL_NAME = "absa-distilbert"


def configure() -> str:
    """Point MLflow at the local sqlite store and select the experiment.

    The experiment is created explicitly on first use so its artifact_location
    can be set. Left to itself MLflow drops a ./mlruns directory wherever the
    process happened to start, which means artifacts land in a different place
    depending on the working directory you invoked training from. Pinning it
    under ml/ keeps runs findable and the repo root clean.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(REPO_ROOT / "ml" / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return TRACKING_URI


def environment_tags(train_size: int, val_size: int) -> dict[str, Any]:
    """Facts about *this* machine and dataset that params alone would not capture.

    A run that scores 0.54 on an RTX 4060 with transformers 5.14 is not
    self-evidently the same run on someone else's box. Recording the environment
    is the difference between "reproducible in principle" and "diagnosable".
    """
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "train_examples": train_size,
        "val_examples": val_size,
    }


def log_evaluation(
    true_by_aspect: list[list[int]],
    pred_by_aspect: list[list[int]],
    split: str,
) -> None:
    """Log the per-class metrics, the report table and the confusion grid."""
    mlflow.log_metrics(
        {f"{split}_{k}": v for k, v in per_class_metrics(true_by_aspect, pred_by_aspect).items()}
    )
    mlflow.log_text(
        classification_report_text(true_by_aspect, pred_by_aspect),
        f"reports/{split}_classification_report.txt",
    )

    figure = confusion_matrix_figure(true_by_aspect, pred_by_aspect, split)
    mlflow.log_figure(figure, f"confusion_matrices/{split}.png")
    # Figures are not garbage-collected by pyplot; without an explicit close,
    # a multi-run sweep leaks one figure per run until matplotlib warns.
    plt.close(figure)


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


def log_model(model_dir: Path, register: bool = False) -> str | None:
    """Log the saved artifact to this run; optionally register it as a version.

    Logging and registering are separate decisions on purpose. Every run should
    keep its own model — that is what makes a run reproducible. But the Model
    Registry is meant to be a curated shortlist of candidates, and auto-
    registering every experiment turns it into a junk drawer where version 14
    means nothing. Registration is therefore explicit: you register a model when
    you have decided it is worth deploying.

    code_paths lists source files individually rather than the src directory.
    Passing a directory would nest imports one level deeper
    (``from src.model import ...``) and break model.py's own ``from data import``
    — listing files keeps them side by side, exactly as in development.

    All three files are required, and each for a different reason: serving.py
    defines the class the pickle refers to, model.py defines the architecture it
    rebuilds, and data.py supplies the ASPECTS and LABEL_NAMES that model.py
    imports. Omit any one and the failure surfaces only at load time.
    """
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=ABSAModel(),
        artifacts={"model_dir": str(model_dir)},
        code_paths=[
            str(SRC_DIR / "serving.py"),
            str(SRC_DIR / "model.py"),
            str(SRC_DIR / "data.py"),
        ],
        registered_model_name=REGISTERED_MODEL_NAME if register else None,
    )

    if not register:
        print(f"logged model to run (not registered): {info.model_uri}")
        return None

    version = str(info.registered_model_version)
    print(f"registered {REGISTERED_MODEL_NAME} version {version}")
    print(f"  load with: mlflow.pyfunc.load_model('models:/{REGISTERED_MODEL_NAME}/{version}')")
    return version
