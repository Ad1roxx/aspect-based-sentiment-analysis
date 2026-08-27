"""Loss behaviour, tested at the tensor level.

compute_loss takes tensors, not a model, so the masking and weighting logic can be
verified without loading DistilBERT. That is worth noticing: the part most likely to
be subtly wrong is also the part that needs no 265 MB checkpoint to test.
"""

from __future__ import annotations

import math

import pytest
import torch

from data import ABSENT, IGNORE_INDEX, NEGATIVE, NUM_CLASSES, POSITIVE
from model import compute_loss

NUM_ASPECTS = 5


def uniform_logits(batch: int = 2) -> torch.Tensor:
    """All-zero logits — the softmax is uniform over the four classes."""
    return torch.zeros(batch, NUM_ASPECTS, NUM_CLASSES)


class TestUntrainedLoss:
    def test_uniform_logits_give_ln_four(self):
        """A model with no preference should score ln(4) ~= 1.3863.

        This is the check that caught nothing in Sprint 1 but would have caught a
        wrong reshape: if the (batch, aspect, class) view were transposed, the loss
        would not land on ln(4).
        """
        labels = torch.zeros(2, NUM_ASPECTS, dtype=torch.long)
        loss = compute_loss(uniform_logits(), labels)
        assert loss.item() == pytest.approx(math.log(NUM_CLASSES), abs=1e-5)


class TestIgnoreIndexMasking:
    def test_masked_positions_do_not_contribute(self):
        """Loss over a batch with one real label must equal the loss over that
        label alone, no matter how many masked positions surround it."""
        logits = torch.randn(1, NUM_ASPECTS, NUM_CLASSES)

        all_but_one = torch.full((1, NUM_ASPECTS), IGNORE_INDEX, dtype=torch.long)
        all_but_one[0, 0] = POSITIVE

        single = torch.nn.functional.cross_entropy(
            logits[0, 0].unsqueeze(0), torch.tensor([POSITIVE])
        )
        assert compute_loss(logits, all_but_one).item() == pytest.approx(
            single.item(), abs=1e-6
        )

    def test_masking_changes_the_result(self):
        """Sanity: if masking made no difference the test above would be vacuous."""
        logits = torch.randn(2, NUM_ASPECTS, NUM_CLASSES)
        labels = torch.full((2, NUM_ASPECTS), ABSENT, dtype=torch.long)
        masked = labels.clone()
        masked[0, 0] = IGNORE_INDEX
        assert compute_loss(logits, labels).item() != pytest.approx(
            compute_loss(logits, masked).item(), abs=1e-9
        )

    def test_all_masked_yields_nan_not_silent_zero(self):
        """Every position masked means no supervision at all.

        PyTorch returns nan here rather than 0.0, and that is the better behaviour:
        a silent zero would look like a perfectly trained batch.
        """
        logits = uniform_logits(1)
        labels = torch.full((1, NUM_ASPECTS), IGNORE_INDEX, dtype=torch.long)
        assert math.isnan(compute_loss(logits, labels).item())


class TestClassWeights:
    def test_weights_change_the_loss(self):
        logits = torch.randn(4, NUM_ASPECTS, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (4, NUM_ASPECTS))
        weights = torch.tensor([0.34, 1.27, 1.61, 0.78])
        assert compute_loss(logits, labels).item() != pytest.approx(
            compute_loss(logits, labels, weights).item(), abs=1e-6
        )

    def test_uniform_weights_match_unweighted(self):
        """Weighting every class by 1.0 must be a no-op.

        cross_entropy normalises by the sum of weights, so this only holds when the
        weights are all exactly 1 — which is the point of checking it.
        """
        logits = torch.randn(4, NUM_ASPECTS, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (4, NUM_ASPECTS))
        ones = torch.ones(NUM_CLASSES)
        assert compute_loss(logits, labels, ones).item() == pytest.approx(
            compute_loss(logits, labels).item(), abs=1e-6
        )

    def test_upweighting_a_rare_class_raises_the_batch_loss(self):
        """The whole point of class weights: getting a rare class wrong hurts more.

        This needs a batch of at least two differently-labelled examples. Weighted
        cross-entropy divides by the summed weights of the present targets, so with
        a single example the normalisation cancels exactly and the weight has no
        visible effect — a real trap when hand-checking this.

        Here: one 'absent' target predicted correctly (low loss) and one 'negative'
        target predicted wrongly (high loss). Upweighting 'negative' shifts the
        average toward the expensive one.
        """
        logits = torch.zeros(2, 1, NUM_CLASSES)
        logits[0, 0, ABSENT] = 5.0    # target absent   -> confident and right
        logits[1, 0, ABSENT] = 5.0    # target negative -> confident and wrong
        labels = torch.tensor([[ABSENT], [NEGATIVE]])

        uniform = compute_loss(logits, labels, torch.ones(NUM_CLASSES))

        weights = torch.ones(NUM_CLASSES)
        weights[NEGATIVE] = 4.0
        weighted = compute_loss(logits, labels, weights)

        assert weighted.item() > uniform.item()
