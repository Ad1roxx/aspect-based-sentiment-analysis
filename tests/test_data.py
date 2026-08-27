"""Parsing and label-mapping tests. No model, no network."""

from __future__ import annotations

import pytest

from data import (
    ABSENT,
    ASPECTS,
    IGNORE_INDEX,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    class_counts,
    parse_semeval_xml,
    train_val_split,
)


class TestParsing:
    def test_parses_every_sentence(self, sample_xml):
        assert len(parse_semeval_xml(sample_xml)) == 4

    def test_multi_aspect_sentence(self, sample_xml):
        example = parse_semeval_xml(sample_xml)[0]
        # (food, service, ambiance, price, misc)
        assert example.labels == (POSITIVE, NEGATIVE, ABSENT, ABSENT, ABSENT)
        assert example.mentioned_aspects() == ["food", "service"]

    def test_unmentioned_aspects_default_to_absent(self, sample_xml):
        example = parse_semeval_xml(sample_xml)[0]
        assert example.labels[ASPECTS.index("ambiance")] == ABSENT

    def test_label_vector_length_matches_aspects(self, sample_xml):
        assert all(len(e.labels) == len(ASPECTS) for e in parse_semeval_xml(sample_xml))


class TestCategoryAliases:
    """Upstream spells two categories differently from this project."""

    def test_ambience_maps_to_ambiance(self, sample_xml):
        example = parse_semeval_xml(sample_xml)[1]
        assert example.labels[ASPECTS.index("ambiance")] == POSITIVE

    def test_anecdotes_maps_to_misc(self, sample_xml):
        example = parse_semeval_xml(sample_xml)[2]
        assert example.labels[ASPECTS.index("misc")] == NEUTRAL

    def test_unknown_category_raises(self, unknown_category_xml):
        # Strict on purpose: a silently dropped category would be an aspect the
        # model never learns, with nothing in the logs to say so.
        with pytest.raises(ValueError, match="unexpected category"):
            parse_semeval_xml(unknown_category_xml)


class TestConflictHandling:
    def test_conflict_becomes_ignore_index(self, sample_xml):
        """'conflict' must be masked, never folded into 'absent'.

        Folding it into absent would teach the model that a clearly discussed
        aspect was never mentioned — actively wrong, not merely lossy.
        """
        example = parse_semeval_xml(sample_xml)[3]
        assert example.labels[ASPECTS.index("food")] == IGNORE_INDEX

    def test_conflict_excluded_from_mentioned_aspects(self, sample_xml):
        assert parse_semeval_xml(sample_xml)[3].mentioned_aspects() == []


class TestPhaseBTrap:
    """Regression test for the Sprint 1 mistake.

    Phase B files have polarity attributes stripped, so every label silently
    becomes None. Training on one produces a model that learns nothing while
    reporting plausible-looking numbers.
    """

    def test_missing_polarity_raises_with_a_useful_message(self, phase_b_xml):
        with pytest.raises(ValueError, match="phase B"):
            parse_semeval_xml(phase_b_xml)

    def test_error_names_the_file(self, phase_b_xml):
        with pytest.raises(ValueError, match="Restaurants_Test_Data_phaseB.xml"):
            parse_semeval_xml(phase_b_xml)


class TestMissingFile:
    def test_missing_file_suggests_the_download_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            parse_semeval_xml(tmp_path / "nope.xml")


class TestSplit:
    def test_split_is_deterministic_for_a_seed(self, examples):
        first, _ = train_val_split(examples, val_fraction=0.5, seed=42)
        second, _ = train_val_split(examples, val_fraction=0.5, seed=42)
        assert [e.sentence_id for e in first] == [e.sentence_id for e in second]

    def test_different_seeds_give_different_splits(self, examples):
        a, _ = train_val_split(examples, val_fraction=0.5, seed=1)
        b, _ = train_val_split(examples, val_fraction=0.5, seed=999)
        assert [e.sentence_id for e in a] != [e.sentence_id for e in b]

    def test_split_partitions_without_loss_or_overlap(self, examples):
        train, val = train_val_split(examples, val_fraction=0.5, seed=42)
        train_ids = {e.sentence_id for e in train}
        val_ids = {e.sentence_id for e in val}
        assert not (train_ids & val_ids)
        assert train_ids | val_ids == {e.sentence_id for e in examples}

    def test_does_not_mutate_input_order(self, examples):
        before = [e.sentence_id for e in examples]
        train_val_split(examples, seed=42)
        assert [e.sentence_id for e in examples] == before


class TestClassCounts:
    def test_counts_every_aspect_slot(self, examples):
        # 6 examples x 5 aspects = 30 slots, minus 1 masked as conflict.
        assert sum(class_counts(examples)) == 29

    def test_ignore_index_is_excluded(self, examples):
        """IGNORE_INDEX contributes nothing to the loss, so letting it into the
        weight calculation would skew every weight."""
        counts = class_counts(examples)
        assert len(counts) == 4
        assert counts[ABSENT] == 24
        assert counts[POSITIVE] == 2
        assert counts[NEGATIVE] == 2
        assert counts[NEUTRAL] == 1
