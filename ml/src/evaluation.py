"""Computing evaluation artifacts: per-class metrics, report tables, confusion grids.

Deliberately separate from tracking.py. These functions *compute* things about a
model's predictions; tracking.py *records* them to MLflow. Keeping the two apart
means evaluate.py can score a saved artifact without MLflow being involved at
all, and the confusion-matrix logic has exactly one home rather than being
reimplemented wherever it is needed.

Nothing here imports mlflow. That is the point.
"""

from __future__ import annotations

import matplotlib

# Select the non-interactive backend before pyplot is imported. Evaluation may
# run over SSH or inside a container where no display exists; without this,
# matplotlib tries to open a GUI window and dies.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)

from data import ASPECTS, LABEL_NAMES  # noqa: E402


def per_class_metrics(
    true_by_aspect: list[list[int]],
    pred_by_aspect: list[list[int]],
) -> dict[str, float]:
    """Flatten per-aspect, per-class F1 into scalar metrics.

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

    Meant to be read by a human deciding what to fix next, not plotted across
    runs — which is why it is an artifact rather than a pile of metrics.
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

    The y-axis labels carry the support count, because normalisation lies at low
    support: a class with a single example that happens to be misrouted renders
    as a solid 100% cell, which looks like confident behaviour rather than one
    data point. Showing n alongside the label keeps that visible.
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
        axis.set_yticklabels(
            [
                f"{name} (n={int(total)})"
                for name, total in zip(LABEL_NAMES, row_totals.flatten())
            ],
            fontsize=8,
        )
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
