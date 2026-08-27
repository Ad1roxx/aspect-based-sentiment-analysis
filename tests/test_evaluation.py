"""Metric computation and report formatting. No model involved."""

from __future__ import annotations

import pytest

from data import ABSENT, ASPECTS, LABEL_NAMES, NEGATIVE, NEUTRAL, POSITIVE
from evaluation import (
    classification_report_text,
    confusion_matrix_figure,
    per_class_metrics,
)


def perfect_predictions():
    """One aspect predicted perfectly, the rest trivially correct."""
    true = [[ABSENT, NEGATIVE, NEUTRAL, POSITIVE] for _ in ASPECTS]
    return true, [list(row) for row in true]


class TestPerClassMetrics:
    def test_perfect_predictions_score_one(self):
        true, pred = perfect_predictions()
        metrics = per_class_metrics(true, pred)
        for aspect in ASPECTS:
            for name in LABEL_NAMES:
                assert metrics[f"f1_{aspect}_{name}"] == pytest.approx(1.0)

    def test_reports_support_alongside_f1(self):
        """F1 of 0.000 over eleven examples and over eleven hundred are different
        facts. Support is logged so nobody reads one as the other."""
        true, pred = perfect_predictions()
        metrics = per_class_metrics(true, pred)
        assert metrics["support_food_neutral"] == 1.0

    def test_absent_class_never_predicted_scores_zero_not_error(self):
        """The real failure mode from Sprint 2: a class the model never predicts.

        zero_division=0 must turn that into 0.0 rather than a warning or a nan,
        because a nan would poison the macro average.
        """
        true = [[NEUTRAL, NEUTRAL] for _ in ASPECTS]
        pred = [[ABSENT, ABSENT] for _ in ASPECTS]
        metrics = per_class_metrics(true, pred)
        assert metrics["f1_food_neutral"] == 0.0

    def test_covers_every_aspect_and_class(self):
        true, pred = perfect_predictions()
        metrics = per_class_metrics(true, pred)
        # 5 aspects x 4 classes x (f1 + support)
        assert len(metrics) == len(ASPECTS) * len(LABEL_NAMES) * 2


class TestReportText:
    def test_names_every_aspect(self):
        true, pred = perfect_predictions()
        report = classification_report_text(true, pred)
        for aspect in ASPECTS:
            assert aspect in report

    def test_includes_all_four_class_rows(self):
        true, pred = perfect_predictions()
        report = classification_report_text(true, pred)
        for name in LABEL_NAMES:
            assert name in report

    def test_reports_a_macro_row_per_aspect(self):
        true, pred = perfect_predictions()
        assert classification_report_text(true, pred).count("macro") == len(ASPECTS)


class TestConfusionFigure:
    def test_builds_a_figure_with_one_axis_per_aspect_plus_a_hidden_spare(self):
        true, pred = perfect_predictions()
        figure = confusion_matrix_figure(true, pred, "test")
        try:
            assert len(figure.axes) == 6  # 2x3 grid, five used, one hidden
        finally:
            figure.clf()

    def test_axis_labels_carry_support_counts(self):
        """Row-normalised colour misleads at low support, so n is shown alongside
        the label. A class with one example must not render as a confident 100%."""
        true, pred = perfect_predictions()
        figure = confusion_matrix_figure(true, pred, "test")
        try:
            labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
            assert all("n=" in label for label in labels)
        finally:
            figure.clf()

    def test_survives_a_class_with_no_examples(self):
        """A split where a class never appears gives a zero row sum. Without the
        clamp in the normaliser that is a divide-by-zero and the figure is nan."""
        true = [[ABSENT, POSITIVE] for _ in ASPECTS]   # no negative, no neutral
        pred = [[ABSENT, POSITIVE] for _ in ASPECTS]
        figure = confusion_matrix_figure(true, pred, "sparse")
        try:
            assert len(figure.axes) == 6
        finally:
            figure.clf()
