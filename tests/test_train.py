"""Class-weight derivation. Pure arithmetic over a label distribution."""

from __future__ import annotations

import pytest
import torch

from data import ABSENT, NEGATIVE, NEUTRAL, POSITIVE, Example, class_counts
from train import class_weight_tensor

CPU = torch.device("cpu")


class TestSchemes:
    def test_none_returns_none(self, examples):
        assert class_weight_tensor(examples, "none", CPU) is None

    def test_unknown_scheme_raises(self, examples):
        with pytest.raises(ValueError, match="unknown scheme"):
            class_weight_tensor(examples, "magic", CPU)

    def test_inverse_matches_the_formula(self, examples):
        """w_c = N / (C * n_c) — the textbook balanced weighting."""
        counts = class_counts(examples)          # [24, 2, 2, 1]
        total = sum(counts)                      # 29
        weights = class_weight_tensor(examples, "inverse", CPU)

        for index, count in enumerate(counts):
            assert weights[index].item() == pytest.approx(total / (4 * count), rel=1e-5)

    def test_sqrt_inverse_is_normalised_to_mean_one(self, examples):
        """Renormalising keeps the loss on roughly the same scale as unweighted,
        so the learning rate does not need retuning alongside the weights."""
        weights = class_weight_tensor(examples, "sqrt-inverse", CPU)
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_sqrt_inverse_is_gentler_than_inverse(self, examples):
        """The Sprint 3 result in one assertion: same direction, less force.

        Full inverse weighting scored *below* the unweighted baseline; the damped
        variant beat it. The spread between the largest and smallest weight is what
        that difference amounts to.
        """
        inverse = class_weight_tensor(examples, "inverse", CPU)
        damped = class_weight_tensor(examples, "sqrt-inverse", CPU)

        inverse_spread = (inverse.max() / inverse.min()).item()
        damped_spread = (damped.max() / damped.min()).item()
        assert damped_spread < inverse_spread

    def test_rare_classes_weigh_more_than_common_ones(self, examples):
        """Ordering is the property that actually matters, and it must hold for
        both schemes: 'absent' is the most common label, so it must weigh least."""
        for scheme in ("inverse", "sqrt-inverse"):
            weights = class_weight_tensor(examples, scheme, CPU)
            assert weights[ABSENT] < weights[POSITIVE] < weights[NEUTRAL]
            assert weights[ABSENT] == weights.min()


class TestDegenerateInput:
    def test_unrepresented_class_raises_rather_than_dividing_by_zero(self):
        """A class with zero examples would give an infinite weight, which silently
        produces nan losses several minutes into training. Fail at setup instead."""
        only_absent = [
            Example("1", "Nothing relevant here.", (ABSENT,) * 5),
        ]
        with pytest.raises(ValueError, match="unrepresented"):
            class_weight_tensor(only_absent, "inverse", CPU)


class TestTensorProperties:
    def test_returns_four_float32_weights_on_the_requested_device(self, examples):
        weights = class_weight_tensor(examples, "sqrt-inverse", CPU)
        assert weights.shape == (4,)
        assert weights.dtype == torch.float32
        assert weights.device.type == "cpu"
