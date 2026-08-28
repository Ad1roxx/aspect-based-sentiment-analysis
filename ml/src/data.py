"""Turn the SemEval-2014 restaurant XML into per-aspect labelled examples.

The task is framed as **five independent 4-way classifications**, one per aspect
category. For a given sentence each aspect is exactly one of:

    absent / negative / neutral / positive

"absent" is the key design choice. Most sentences mention only one or two of the
five categories — 2014 of the 2585 training sentences mention exactly one — so a
plain 3-class sentiment scheme has nowhere to put "this review never talks about
price". Making absence its own class lets one forward pass answer both "is this
aspect discussed?" and "how does the writer feel about it?".

Run directly to inspect the parsed corpus::

    python ml/src/data.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
TRAIN_XML = RAW_DIR / "Restaurants_Train_v2.xml"
TEST_XML = RAW_DIR / "Restaurants_Test_Gold.xml"

# The five aspect categories this project predicts, in a fixed order. Every
# label vector follows this order, so it must not be reshuffled once a model is
# trained — index 0 means "food" in the data, the model and the API alike.
ASPECTS: tuple[str, ...] = ("food", "service", "ambience", "price", "misc")
ASPECT_INDEX: dict[str, int] = {name: i for i, name in enumerate(ASPECTS)}

# One upstream category is renamed. "anecdotes/miscellaneous" is unusable as an
# API field name and as a UI label, so it becomes "misc". Mapping it explicitly
# — rather than quietly rewriting during parsing — keeps the translation auditable.
#
# The other four categories are used exactly as SemEval spells them, including
# "ambience". An earlier version aliased that to "ambiance"; the alias was removed
# because renaming a dataset field for no reason is a thing you then have to
# explain forever.
CATEGORY_ALIASES: dict[str, str] = {
    "anecdotes/miscellaneous": "misc",
}

ABSENT, NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2, 3
LABEL_NAMES: tuple[str, ...] = ("absent", "negative", "neutral", "positive")
NUM_CLASSES = len(LABEL_NAMES)

POLARITY_TO_LABEL: dict[str, int] = {
    "negative": NEGATIVE,
    "neutral": NEUTRAL,
    "positive": POSITIVE,
}

# PyTorch's cross-entropy loss skips targets equal to -100 by default. We reuse
# that convention for "conflict" annotations (see _label_for_polarity).
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Example:
    """One sentence and its label for each of the five aspects."""

    sentence_id: str
    text: str
    labels: tuple[int, ...]  # len == len(ASPECTS)

    def mentioned_aspects(self) -> list[str]:
        return [
            name
            for name, label in zip(ASPECTS, self.labels)
            if label not in (ABSENT, IGNORE_INDEX)
        ]


def _label_for_polarity(polarity: str | None, source: Path) -> int:
    """Map an upstream polarity string onto our label space.

    Two upstream values need deliberate handling:

    ``conflict``
        ~5% of annotations, where the sentence is both positive and negative
        about one aspect. This project's label space has no room for it. Rather
        than discard the whole sentence — which would also throw away its other,
        perfectly usable aspects — we mark just that aspect with IGNORE_INDEX so
        the loss function skips it. The alternative, folding conflict into
        "absent", would be actively harmful: it would teach the model that a
        clearly discussed aspect was never mentioned.

    ``None``
        Means the file is a "phase B" release, where polarities were stripped
        because competitors were given the gold aspects and asked to predict
        them. Such a file is unusable as training or test data, so fail loudly.
    """
    if polarity == "conflict":
        return IGNORE_INDEX

    if polarity is None:
        raise ValueError(
            f"{source.name} has aspectCategory entries with no polarity attribute.\n"
            "This is the signature of a 'phase B' file, whose labels were removed "
            "on purpose. Use Restaurants_Test_Gold.xml instead."
        )

    try:
        return POLARITY_TO_LABEL[polarity]
    except KeyError:
        raise ValueError(f"unrecognised polarity {polarity!r} in {source.name}") from None


def parse_semeval_xml(path: Path) -> list[Example]:
    """Parse one SemEval XML file into Examples.

    Only ``aspectCategories`` is read. The files also carry ``aspectTerms`` — the
    explicit spans such as "waiter" — but those drive aspect *term* extraction, a
    different task. This project predicts a fixed set of abstract categories, so
    the term annotations are deliberately unused.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )

    root = ET.parse(path).getroot()
    examples: list[Example] = []

    for sentence in root.findall("sentence"):
        text = (sentence.findtext("text") or "").strip()
        if not text:
            continue

        labels = [ABSENT] * len(ASPECTS)

        for node in sentence.findall("./aspectCategories/aspectCategory"):
            raw_category = node.get("category")
            category = CATEGORY_ALIASES.get(raw_category, raw_category)

            if category not in ASPECT_INDEX:
                raise ValueError(
                    f"unexpected category {raw_category!r} in {path.name}. "
                    f"Known categories: {sorted(ASPECT_INDEX)}"
                )

            labels[ASPECT_INDEX[category]] = _label_for_polarity(
                node.get("polarity"), path
            )

        examples.append(
            Example(
                sentence_id=sentence.get("id", ""),
                text=text,
                labels=tuple(labels),
            )
        )

    return examples


def train_val_split(
    examples: list[Example],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Split off a validation set with a seeded shuffle.

    This is a plain random split, not a stratified one. Stratifying would mean
    grouping on the combination of five aspect labels, and many of those
    combinations occur only once or twice — too rare to divide between splits.
    The distributions are printed by ``describe`` so the split can be checked
    rather than assumed.
    """
    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def class_counts(examples: list[Example]) -> list[int]:
    """Count each label across every (sentence, aspect) pair in a split.

    Counted globally rather than per aspect, because this feeds a single set of
    class weights into one flattened cross-entropy call. Per-aspect weights would
    be more precise — 'neutral' is far rarer for price than for misc — but would
    require five separate loss computations. Noted as a limitation rather than
    silently ignored.

    IGNORE_INDEX positions are excluded: they contribute nothing to the loss, so
    letting them skew the weights would be a bug.
    """
    counts = Counter(
        label
        for example in examples
        for label in example.labels
        if label != IGNORE_INDEX
    )
    return [counts.get(label, 0) for label in range(NUM_CLASSES)]


def describe(name: str, examples: list[Example]) -> None:
    """Print the per-aspect label distribution for one split."""
    print(f"\n{name}  ({len(examples)} sentences)")
    header = f"  {'aspect':<10}" + "".join(f"{n:>10}" for n in LABEL_NAMES) + f"{'conflict':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, aspect in enumerate(ASPECTS):
        counts = Counter(example.labels[i] for example in examples)
        row = f"  {aspect:<10}"
        row += "".join(f"{counts.get(label, 0):>10}" for label in range(NUM_CLASSES))
        row += f"{counts.get(IGNORE_INDEX, 0):>10}"
        print(row)

    per_sentence = Counter(len(example.mentioned_aspects()) for example in examples)
    print(f"  aspects per sentence: {dict(sorted(per_sentence.items()))}")


def load_splits(
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Load train / validation / test.

    The test set is the untouched official gold file, so results stay comparable
    with published SemEval numbers. Validation is carved out of training only —
    the test set is never used for tuning.
    """
    train_all = parse_semeval_xml(TRAIN_XML)
    test = parse_semeval_xml(TEST_XML)
    train, val = train_val_split(train_all, val_fraction=val_fraction, seed=seed)
    return train, val, test


def main() -> None:
    train, val, test = load_splits()

    for name, split in (("TRAIN", train), ("VALIDATION", val), ("TEST", test)):
        describe(name, split)

    print("\nSample examples:")
    for example in train[:3]:
        mentioned = {
            aspect: LABEL_NAMES[example.labels[ASPECT_INDEX[aspect]]]
            for aspect in example.mentioned_aspects()
        }
        print(f"  {example.text}")
        print(f"    -> {mentioned}")


if __name__ == "__main__":
    main()
