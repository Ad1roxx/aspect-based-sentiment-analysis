"""Subword merging — the readable-output half of the explainability method.

The attribution maths needs a real model and lives in the integration tests. This
part is pure string and arithmetic handling, and it is where the bugs actually are.
"""

from __future__ import annotations

import pytest

from explain import SPECIAL_TOKENS, merge_subwords


class TestMerging:
    def test_wordpiece_fragments_recombine(self):
        """DistilBERT splits 'overpriced' into three pieces. Displaying them
        separately is unreadable AND splits one word's contribution three ways, so
        each fragment looks less important than the word actually was."""
        tokens = ["over", "##pric", "##ed"]
        words, scores = merge_subwords(tokens, [0.2, 0.3, 0.5])
        assert words == ["overpriced"]

    def test_fragment_scores_are_summed_not_averaged(self):
        """Attributions are additive in the embedding dimension, so summing is the
        correct aggregation. Averaging would systematically under-rank long words."""
        _, scores = merge_subwords(["over", "##pric", "##ed"], [0.2, 0.3, 0.5])
        assert scores == [pytest.approx(1.0)]

    def test_signed_fragments_cancel_correctly(self):
        """Summing happens while scores are still signed, before magnitude is taken.
        Taking absolute values first would let fragments inflate each other."""
        _, scores = merge_subwords(["un", "##happy"], [-0.7, 0.2])
        assert scores == [pytest.approx(-0.5)]

    def test_whole_words_pass_through_untouched(self):
        words, scores = merge_subwords(["the", "food"], [0.1, 0.9])
        assert words == ["the", "food"]
        assert scores == [pytest.approx(0.1), pytest.approx(0.9)]

    def test_mixed_sequence(self):
        tokens = ["the", "food", "was", "over", "##pric", "##ed"]
        words, scores = merge_subwords(tokens, [0.1, 0.2, 0.1, 0.2, 0.2, 0.2])
        assert words == ["the", "food", "was", "overpriced"]
        assert scores[-1] == pytest.approx(0.6)


class TestSpecialTokens:
    def test_special_tokens_are_dropped(self):
        """[CLS] accrues large attribution simply for being the pooled position,
        which says nothing about the input text."""
        tokens = ["[CLS]", "great", "food", "[SEP]"]
        words, _ = merge_subwords(tokens, [9.9, 0.5, 0.4, 9.9])
        assert words == ["great", "food"]

    def test_every_declared_special_token_is_dropped(self):
        words, _ = merge_subwords(list(SPECIAL_TOKENS), [1.0] * len(SPECIAL_TOKENS))
        assert words == []

    def test_fragment_after_a_special_token_starts_a_new_word(self):
        """A '##' fragment with no preceding word must not crash or silently attach
        itself to a dropped [CLS]."""
        words, _ = merge_subwords(["[CLS]", "##ed"], [0.1, 0.2])
        assert words == ["##ed"]


class TestEdgeCases:
    def test_empty_input(self):
        assert merge_subwords([], []) == ([], [])

    def test_only_special_tokens(self):
        assert merge_subwords(["[CLS]", "[SEP]"], [1.0, 1.0]) == ([], [])

    def test_output_lists_stay_aligned(self):
        tokens = ["[CLS]", "a", "##b", "c", "[SEP]"]
        words, scores = merge_subwords(tokens, [0.0, 0.1, 0.2, 0.3, 0.0])
        assert len(words) == len(scores)
