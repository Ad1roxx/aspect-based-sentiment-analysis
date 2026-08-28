"""Shared fixtures.

The organising problem for this suite: the trained artifact is 265 MB, gitignored,
and produced on a GPU. CI has none of those. A test suite that can only run on the
machine that trained the model is not a test suite.

So the tests are layered:

  * **pure logic** — parsing, label mapping, loss masking, metrics, subword merging.
    No model, no network, no artifact. These are the majority and they run anywhere.
  * **API contract** — routes, validation, status codes, response shape. Driven
    through a *fake* service, so they test the web layer rather than the model.
  * **integration** — marked ``integration``, needs the real artifact, and skips
    cleanly when it is absent. These run locally; CI skips them.

The split matters beyond convenience: a test that loads a 265 MB checkpoint to check
that empty input returns 422 is testing the wrong thing anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data import ABSENT, IGNORE_INDEX, NEGATIVE, NEUTRAL, POSITIVE, Example

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "ml" / "models" / "absa-distilbert"

# One SemEval sentence per interesting case, hand-written rather than sampled from
# the real corpus: a fixture you can read is worth more than a realistic one, and
# these must stay stable even if the dataset is re-downloaded.
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sentences>
    <sentence id="1">
        <text>The pasta was incredible but the waiter ignored us.</text>
        <aspectCategories>
            <aspectCategory category="food" polarity="positive"/>
            <aspectCategory category="service" polarity="negative"/>
        </aspectCategories>
    </sentence>
    <sentence id="2">
        <text>Nice ambience, though a bit pricey.</text>
        <aspectCategories>
            <aspectCategory category="ambience" polarity="positive"/>
            <aspectCategory category="price" polarity="negative"/>
        </aspectCategories>
    </sentence>
    <sentence id="3">
        <text>I have been going there for years.</text>
        <aspectCategories>
            <aspectCategory category="anecdotes/miscellaneous" polarity="neutral"/>
        </aspectCategories>
    </sentence>
    <sentence id="4">
        <text>The food was great but also terrible.</text>
        <aspectCategories>
            <aspectCategory category="food" polarity="conflict"/>
        </aspectCategories>
    </sentence>
</sentences>
"""

# A phase B file: aspect categories present, polarity attributes stripped. This is
# the trap Sprint 1 hit, so it gets a regression test.
PHASE_B_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sentences>
    <sentence id="1">
        <text>The pasta was incredible.</text>
        <aspectCategories>
            <aspectCategory category="food"/>
        </aspectCategories>
    </sentence>
</sentences>
"""

UNKNOWN_CATEGORY_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sentences>
    <sentence id="1">
        <text>The parking was awful.</text>
        <aspectCategories>
            <aspectCategory category="parking" polarity="negative"/>
        </aspectCategories>
    </sentence>
</sentences>
"""


@pytest.fixture
def sample_xml(tmp_path: Path) -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")
    return path


@pytest.fixture
def phase_b_xml(tmp_path: Path) -> Path:
    path = tmp_path / "Restaurants_Test_Data_phaseB.xml"
    path.write_text(PHASE_B_XML, encoding="utf-8")
    return path


@pytest.fixture
def unknown_category_xml(tmp_path: Path) -> Path:
    path = tmp_path / "unknown.xml"
    path.write_text(UNKNOWN_CATEGORY_XML, encoding="utf-8")
    return path


@pytest.fixture
def examples() -> list[Example]:
    """A small labelled set with a deliberately skewed class distribution.

    Label order is (food, service, ambience, price, misc) — the ASPECTS order, which
    is load-bearing everywhere.
    """
    return [
        Example("1", "The pasta was incredible.", (POSITIVE, ABSENT, ABSENT, ABSENT, ABSENT)),
        Example("2", "The waiter was rude.", (ABSENT, NEGATIVE, ABSENT, ABSENT, ABSENT)),
        Example("3", "Lovely room.", (ABSENT, ABSENT, POSITIVE, ABSENT, ABSENT)),
        Example("4", "Too expensive.", (ABSENT, ABSENT, ABSENT, NEGATIVE, ABSENT)),
        Example("5", "Been going for years.", (ABSENT, ABSENT, ABSENT, ABSENT, NEUTRAL)),
        Example("6", "Great and awful food.", (IGNORE_INDEX, ABSENT, ABSENT, ABSENT, ABSENT)),
    ]


artifact_required = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.pt").is_file(),
    reason=f"no trained artifact at {ARTIFACT_DIR} — run ml/src/train.py",
)
