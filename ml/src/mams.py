"""MAMS-ACSA as supplementary training data for the multi-aspect failure.

Sprint 6 measured the problem: of aspects genuinely discussed, the model detects
91.8% in single-aspect sentences and 78.8% in multi-aspect ones. The cause is
mostly the training data — 77.9% of SemEval-2014 sentences mention exactly one
aspect, and only 448 mention two or more.

MAMS-ACSA (Jiang et al., EMNLP-IJCNLP 2019) is built for precisely this: every
one of its 3,949 sentences carries at least two aspects with DIFFERENT sentiment
polarities. Verified, not assumed — 3949/3949 satisfy both conditions.

TWO THINGS MAKE THIS NON-TRIVIAL

1. It overlaps our data. MAMS comes from the same CSNY / Citysearch corpus as
   SemEval. 67 MAMS sentences already appear in ours, 11 of them in our TEST
   split. Training on those would inflate the test score for the worst possible
   reason. They are removed here.

2. Its 'neutral' is not our 'neutral'. MAMS was constructed so that every
   sentence has at least two differing polarities, and 'neutral' absorbed the
   slack. Measured:

       neutral rate, SemEval-2014 (mentioned aspects)   13%
       neutral rate, MAMS                               43%
       ... 'menu' 79%, 'place' 60%, 'food' 57%

   That is a different label doing a different job — closer to "mentioned, no
   strong opinion" than to genuine neutral sentiment. Merging it directly would
   teach the model two incompatible meanings for one class.

Hence two modes, so the difference can be measured rather than argued about:

``filtered``  MAMS positive/negative are kept; MAMS neutral is masked with
              IGNORE_INDEX so it contributes nothing to the loss. Unmentioned
              aspects become ABSENT, which is where the detection signal comes
              from — the model sees "here food is positive AND service is
              negative AND price is absent".
``full``      everything kept, neutral included. The naive merge, included so
              the claim above is testable instead of asserted.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from data import (
    ABSENT,
    ASPECT_INDEX,
    ASPECTS,
    IGNORE_INDEX,
    POLARITY_TO_LABEL,
    Example,
    load_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MAMS_FILES = (
    RAW_DIR / "mams_acsa_train.xml",
    RAW_DIR / "mams_acsa_val.xml",
    RAW_DIR / "mams_acsa_test.xml",
)

MODES = ("none", "filtered", "full")

# MAMS uses eight categories; this project uses five. Two of the merges are
# judgement calls and are flagged as such rather than presented as obvious:
#
#   staff  -> service    safe. "the hostess", "our waitress" is service.
#   menu   -> food       shaky. 79% of 'menu' annotations are neutral and many
#                        are really about a waiter knowing the menu, i.e.
#                        service. Kept with food because the menu describes what
#                        is served, but this is the weakest link in the mapping.
#   place  -> ambience   mixed. "impressed by the room" is ambience; "just the
#                        place for you" is closer to misc. Sampled both.
CATEGORY_MAP: dict[str, str] = {
    "food": "food",
    "menu": "food",
    "service": "service",
    "staff": "service",
    "ambience": "ambience",
    "place": "ambience",
    "price": "price",
    "miscellaneous": "misc",
}


def normalise(text: str) -> str:
    """Aggressive normalisation for overlap detection.

    Lowercase, strip everything but alphanumerics and spaces, collapse runs of
    whitespace. Deliberately lossy: the goal is to catch the same sentence
    re-typed or re-punctuated across two corpora, and a conservative comparison
    would miss those and let them through into training.
    """
    text = text.lower()
    # Apostrophes are DELETED, other punctuation becomes a space. The difference
    # matters: replacing an apostrophe with a space turns "it's" into "it s",
    # which then never matches a corpus that wrote "its" — while "good.Bad" must
    # still split into two words rather than fusing.
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_labels(
    annotations: list[tuple[str, str]],
    mode: str,
) -> tuple[tuple[int, ...], Counter]:
    """Map one sentence's MAMS annotations onto this project's label vector.

    Pure, so it can be unit-tested with synthetic input — the data files are
    gitignored and absent in CI, and this is where the subtle decisions live.

    Returns the label tuple plus a Counter of what had to be masked and why.
    """
    labels = [ABSENT] * len(ASPECTS)
    stats: Counter = Counter()

    for category, polarity in annotations:
        mapped = CATEGORY_MAP.get(category)
        if mapped is None:
            raise ValueError(f"unmapped MAMS category {category!r}")

        if polarity == "neutral" and mode == "filtered":
            # Masked, not dropped. The sentence keeps its other aspects, and this
            # position contributes nothing to the loss — the same mechanism used
            # for SemEval 'conflict'.
            label = IGNORE_INDEX
            stats["masked_neutral"] += 1
        else:
            label = POLARITY_TO_LABEL[polarity]

        # Two MAMS categories can collapse onto one of ours (staff+service,
        # place+ambience). If they disagree, the aspect is genuinely conflicted at
        # our granularity, so mask it rather than letting whichever came last
        # silently win — which would make the label depend on XML ordering.
        slot = ASPECT_INDEX[mapped]
        if labels[slot] not in (ABSENT, label):
            labels[slot] = IGNORE_INDEX
            stats["masked_merge_conflict"] += 1
        else:
            labels[slot] = label

    return tuple(labels), stats


def _parse(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )
    root = ET.parse(path).getroot()
    parsed = []
    for sentence in root.findall("sentence"):
        text = (sentence.findtext("text") or "").strip()
        if not text:
            continue
        annotations = [
            (node.get("category"), node.get("polarity"))
            for node in sentence.findall("./aspectCategories/aspectCategory")
        ]
        parsed.append((text, annotations))
    return parsed


def load_mams(mode: str = "filtered", verbose: bool = False) -> list[Example]:
    """Load MAMS as Examples in this project's label space.

    Sentences overlapping ANY of our splits are dropped — not just the test
    split. Overlap with train would duplicate sentences under two different
    annotation schemes, which is its own kind of mess.
    """
    if mode == "none":
        return []
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")

    train, val, test = load_splits()
    ours = {normalise(example.text) for example in train + val + test}

    examples: list[Example] = []
    stats: Counter = Counter()

    for path in MAMS_FILES:
        for index, (text, annotations) in enumerate(_parse(path)):
            if normalise(text) in ours:
                stats["dropped_overlap"] += 1
                continue

            labels, counts = to_labels(annotations, mode)
            stats.update(counts)

            examples.append(
                Example(
                    sentence_id=f"mams-{path.stem}-{index}",
                    text=text,
                    labels=tuple(labels),
                )
            )
            stats["kept"] += 1

    if verbose:
        print(
            f"MAMS ({mode}): kept {stats['kept']}, "
            f"dropped {stats['dropped_overlap']} overlapping ours, "
            f"masked {stats['masked_neutral']} neutral, "
            f"{stats['masked_merge_conflict']} merge conflicts"
        )
    return examples


def main() -> None:
    from data import describe

    train, _, _ = load_splits()
    for mode in ("filtered", "full"):
        extra = load_mams(mode, verbose=True)
        describe(f"MAMS ({mode})", extra)

    base = sum(1 for e in train if len(e.mentioned_aspects()) >= 2)
    added = sum(1 for e in load_mams("filtered") if len(e.mentioned_aspects()) >= 2)
    print("\nmulti-aspect sentences available for training:")
    print(f"  SemEval-2014 alone : {base}")
    print(f"  with MAMS          : {base + added}  ({(base + added) / base:.1f}x)")


if __name__ == "__main__":
    main()
