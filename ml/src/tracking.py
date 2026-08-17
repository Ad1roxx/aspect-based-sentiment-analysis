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

import matplotlib

# Select the non-interactive backend before pyplot is imported. Training may run
# over SSH or inside a container where no display exists; without this,
# matplotlib tries to open a GUI window and dies.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)

from data import ASPECTS, LABEL_NAMES  # noqa: E402
from serving import ABSAModel  # noqa: E402

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


# ---------------------------------------------------------------------------
# Evaluation artifacts
# ---------------------------------------------------------------------------


def per_class_metrics(
    true_by_aspect: list[list[int]],
    pred_by_aspect: list[list[int]],
) -> dict[str, float]:
    """Flatten per-aspect, per-class F1 into MLflow-loggable scalar metrics.

    MLflow metrics are flat key -> float, so the two-dimensional (aspect, class)
    result is encoded in the key: ``f1_food_neutral``. Verbose, but it makes the
    collapse searchable and chartable across runs, which a nested blob would not.
    """
    metrics: dict[str, float] = {}
    for index, aspect in enumerate(ASPECTS):
        _, _, f1, support = precision_recall_fscore_support(
            true_by_aspect[index],
            pred_by_aspect[index],
            labels=list(range(len(LABEL_NAMES))),
            zero_division=0,
        )
        for class_index, class_name in enumerate(LABEL_NAMES):
            metrics[f"f1_{aspect}_{class_name}"] = float(f1[class_index])
            # Support is a property of the data, not the model, but logging it
            # alongside stops anyone reading an F1 of 0.000 without noticing it
            # was computed over eleven examples.
            metrics[f"support_{aspect}_{class_name}"] = float(support[class_index])
    return metrics


def classification_report_text(
    true_by_aspect: list[list[int]],
    pred_by_aspect: list[list[int]],
) -> str:
    """A precision/recall/F1/support table per aspect, as plain text.

    Logged as an artifact rather than metrics because it is meant to be read by
    a human deciding what to fix next, not plotted across runs.
    """
    lines: list[str] = []
    for index, aspect in enumerate(ASPECTS):
        precision, recall, f1, support = precision_recall_fscore_support(
            true_by_aspect[index],
            pred_by_aspect[index],
            labels=list(range(len(LABEL_NAMES))),
            zero_division=0,
        )
        lines.append(f"{aspect}")
        lines.append(f"  {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
        for class_index, class_name in enumerate(LABEL_NAMES):
            lines.append(
                f"  {class_name:<10}"
                f"{precision[class_index]:>8.3f}"
                f"{recall[class_index]:>8.3f}"
                f"{f1[class_index]:>8.3f}"
                f"{support[class_index]:>9d}"
            )
        lines.append(f"  {'macro':<10}{'':>8}{'':>8}{np.mean(f1):>8.3f}")
        lines.append("")
    return "\n".join(lines)


def confusion_matrix_figure(
    true_by_aspect: list[list[int]],
    pred_by_aspect: list[list[int]],
    title: str,
) -> plt.Figure:
    """One confusion matrix per aspect, as a 2x3 grid.

    Cells are ANNOTATED with raw counts but COLOURED by row-normalised
    proportion. That distinction matters: 'absent' outnumbers every sentiment
    class by an order of magnitude, so colouring by raw count would render the
    entire grid one dark square in the top-left corner and four invisible rows.
    Normalising per true-class turns each row into "of the examples that really
    were X, where did they go?" — which is recall, and recall is exactly what
    collapses here.
    """
    rows, columns = 2, 3
    figure, axes = plt.subplots(rows, columns, figsize=(13, 8))
    axes = axes.flatten()

    for index, aspect in enumerate(ASPECTS):
        axis = axes[index]
        matrix = confusion_matrix(
            true_by_aspect[index],
            pred_by_aspect[index],
            labels=list(range(len(LABEL_NAMES))),
        )

        # Row sums of zero (a class absent from this split) would divide by zero;
        # clamp to 1 so the row renders as all-zero rather than NaN.
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalised = matrix / np.maximum(row_totals, 1)

        axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(aspect, fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(LABEL_NAMES)))
        axis.set_yticks(range(len(LABEL_NAMES)))
        axis.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(LABEL_NAMES, fontsize=8)
        axis.set_xlabel("predicted", fontsize=9)
        axis.set_ylabel("true", fontsize=9)

        for row in range(len(LABEL_NAMES)):
            for column in range(len(LABEL_NAMES)):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    # White text on dark cells, black on light, so annotations
                    # stay legible at both ends of the colour scale.
                    color="white" if normalised[row, column] > 0.5 else "black",
                )

    # Five aspects in a six-cell grid leaves one empty; hide it rather than
    # letting an empty pair of axes render.
    for spare in range(len(ASPECTS), rows * columns):
        axes[spare].axis("off")

    figure.suptitle(
        f"{title} — cells are counts, colour is row-normalised (recall)",
        fontsize=12,
    )
    figure.tight_layout()
    return figure


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


def log_model(model_dir: Path) -> str:
    """Log the saved artifact as a registered model and return its version.

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
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    version = str(info.registered_model_version)
    print(f"registered {REGISTERED_MODEL_NAME} version {version}")
    print(f"  load with: mlflow.pyfunc.load_model('models:/{REGISTERED_MODEL_NAME}/{version}')")
    return version
