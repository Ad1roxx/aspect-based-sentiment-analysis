"""MAMS category mapping and label construction. No data files, no model.

The data files are gitignored and absent in CI, so these test `to_labels` — the
pure function holding every judgement call — with synthetic annotations.
"""

from __future__ import annotations

import pytest

from data import ABSENT, ASPECTS, IGNORE_INDEX, NEGATIVE, NEUTRAL, POSITIVE
from mams import (
    CATEGORY_MAP,
    MODES,
    REMAP_TARGETS,
    UNRELIABLE_CATEGORIES,
    load_mams,
    normalise,
    to_labels,
)

FOOD, SERVICE, AMBIENCE, PRICE, MISC = (ASPECTS.index(a) for a in ASPECTS)


class TestCategoryMap:
    def test_covers_every_mams_category(self):
        """MAMS has exactly eight. An unmapped one raises rather than being
        silently dropped, so this list must stay complete."""
        assert set(CATEGORY_MAP) == {
            "food", "menu", "service", "staff",
            "ambience", "place", "price", "miscellaneous",
        }

    def test_every_target_is_one_of_ours(self):
        assert set(CATEGORY_MAP.values()) <= set(ASPECTS)

    def test_unmapped_category_raises(self):
        with pytest.raises(ValueError, match="unmapped MAMS category"):
            to_labels([("drinks", "positive")], "full")


class TestLabelConstruction:
    def test_unmentioned_aspects_become_absent(self):
        """This is where the detection signal comes from: the model learns that
        in a sentence about food and service, price was NOT discussed."""
        labels, _ = to_labels([("food", "positive")], "full")
        assert labels[FOOD] == POSITIVE
        assert labels[SERVICE] == ABSENT
        assert labels[PRICE] == ABSENT

    def test_staff_maps_to_service(self):
        labels, _ = to_labels([("staff", "negative")], "full")
        assert labels[SERVICE] == NEGATIVE

    def test_place_maps_to_ambience(self):
        labels, _ = to_labels([("place", "positive")], "full")
        assert labels[AMBIENCE] == POSITIVE

    def test_menu_maps_to_food(self):
        labels, _ = to_labels([("menu", "negative")], "full")
        assert labels[FOOD] == NEGATIVE


class TestNeutralHandling:
    def test_full_mode_keeps_neutral(self):
        labels, stats = to_labels([("food", "neutral")], "full")
        assert labels[FOOD] == NEUTRAL
        assert stats["masked_neutral"] == 0

    def test_filtered_mode_masks_neutral(self):
        """MAMS's neutral rate is 43% against SemEval's 13% — it is a different
        label doing a different job, so 'filtered' excludes it from the loss."""
        labels, stats = to_labels([("food", "neutral")], "filtered")
        assert labels[FOOD] == IGNORE_INDEX
        assert stats["masked_neutral"] == 1

    def test_filtering_does_not_discard_the_other_aspects(self):
        """Masking is per-aspect, not per-sentence. A sentence with one neutral
        and one negative still contributes the negative."""
        labels, _ = to_labels(
            [("food", "neutral"), ("staff", "negative")], "filtered"
        )
        assert labels[FOOD] == IGNORE_INDEX
        assert labels[SERVICE] == NEGATIVE


class TestMergeConflicts:
    def test_agreeing_categories_merge_cleanly(self):
        """staff and service both map to service; agreeing is not a conflict."""
        labels, stats = to_labels(
            [("staff", "negative"), ("service", "negative")], "full"
        )
        assert labels[SERVICE] == NEGATIVE
        assert stats["masked_merge_conflict"] == 0

    def test_disagreeing_categories_are_masked(self):
        """Two MAMS categories collapsing onto one of ours with different
        polarities is genuinely ambiguous at our granularity."""
        labels, stats = to_labels(
            [("staff", "positive"), ("service", "negative")], "full"
        )
        assert labels[SERVICE] == IGNORE_INDEX
        assert stats["masked_merge_conflict"] == 1

    def test_conflict_resolution_does_not_depend_on_order(self):
        """Without explicit masking, whichever annotation came last in the XML
        would silently win — making labels depend on file ordering."""
        forward, _ = to_labels([("staff", "positive"), ("service", "negative")], "full")
        reverse, _ = to_labels([("service", "negative"), ("staff", "positive")], "full")
        assert forward == reverse


class TestNormalise:
    def test_case_and_punctuation_are_stripped(self):
        assert normalise("The Food, was GREAT!") == "the food was great"

    def test_whitespace_is_collapsed(self):
        assert normalise("a\t b\n\nc") == "a b c"

    def test_matches_across_punctuation_differences(self):
        """The point of the overlap check: the same sentence re-punctuated in
        another corpus must still collide."""
        assert normalise("It's good.") == normalise("Its good")


class TestModes:
    def test_none_returns_nothing(self):
        assert load_mams("none") == []

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            load_mams("everything")

    def test_declared_modes(self):
        assert MODES == ("none", "filtered", "full", "curated", "remapped")


class TestPlaceHandling:
    """MAMS 'place' is 60% neutral against our ambience's 6%, and the model
    learned the literal token — 'Cosy little place.' predicted ambience negative
    while 'Cosy little restaurant.' predicted absent. Two responses, both here."""

    def test_full_mode_sends_place_to_ambience(self):
        labels, _ = to_labels([("place", "neutral")], "full")
        assert labels[AMBIENCE] == NEUTRAL

    def test_curated_mode_masks_place(self):
        labels, stats = to_labels([("place", "neutral")], "curated")
        assert labels[AMBIENCE] == IGNORE_INDEX
        assert stats["masked_unreliable"] == 1

    def test_remapped_mode_sends_place_to_misc(self):
        """Keeps the detection signal instead of discarding it — MAMS place
        (21/60/19) fits our misc (18/33/49) far better than our ambience."""
        labels, stats = to_labels([("place", "neutral")], "remapped")
        assert labels[MISC] == NEUTRAL
        assert labels[AMBIENCE] == ABSENT
        assert stats["remapped"] == 1

    def test_genuine_ambience_is_untouched_in_every_mode(self):
        """Only 'place' is reclassified. MAMS's own 'ambience' category is
        distribution-compatible with ours and must keep flowing through."""
        for mode in ("full", "curated", "remapped"):
            labels, _ = to_labels([("ambience", "positive")], mode)
            assert labels[AMBIENCE] == POSITIVE, mode

    def test_the_two_policies_are_declared_consistently(self):
        assert UNRELIABLE_CATEGORIES <= set(CATEGORY_MAP)
        assert set(REMAP_TARGETS) <= set(CATEGORY_MAP)
        assert set(REMAP_TARGETS.values()) <= set(ASPECTS)
