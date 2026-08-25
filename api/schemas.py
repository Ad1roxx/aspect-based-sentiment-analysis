"""Request and response shapes for the API.

These Pydantic models are the API's contract. They exist separately from the
route handlers because they are the part the React frontend codes against — if a
field name changes here, something breaks in the browser, and that deserves to
be visible in a diff rather than buried inside a handler.

Pydantic also generates the OpenAPI schema from these classes, so the docs at
/docs are derived from the same definitions the server validates against. They
cannot drift apart.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Pydantic v2 reserves the "model_" prefix for its own attributes and warns on
# any field using it. Several fields here are genuinely about the ML model, so
# the protection is disabled rather than renaming them to something evasive like
# "mdl_loaded". Applied per-class below.
ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The review text to analyse.",
        examples=["The pasta was incredible but the waiter ignored us."],
    )
    explain: bool = Field(
        default=False,
        description=(
            "Return per-word importance for each detected aspect. Costs one "
            "extra backward pass per detected aspect, so it is opt-in."
        ),
    )


class WordImportance(BaseModel):
    word: str
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Unsigned attribution, normalised so the strongest word in this "
            "explanation is 1.0. Indicates that a word mattered, NOT which way "
            "it pushed the prediction."
        ),
    )


class AspectPrediction(BaseModel):
    aspect: str
    label: str = Field(..., description="absent | negative | neutral | positive")
    confidence: float = Field(
        ...,
        description=(
            "Softmax probability of the chosen class. This is the model's "
            "relative preference among four options, not a calibrated "
            "probability of being correct."
        ),
    )
    mentioned: bool = Field(
        ...,
        description="False when label == 'absent'. Provided so the UI does not "
        "have to hard-code the string 'absent'.",
    )
    words: list[WordImportance] | None = Field(
        default=None,
        description="Present only when explain=true and the aspect was mentioned.",
    )


class PredictResponse(BaseModel):
    text: str
    aspects: list[AspectPrediction]
    explained: bool
    truncated: bool = Field(
        ...,
        description=(
            "True when the input exceeded the model's token limit and was cut. "
            "Surfaced rather than hidden: a silently truncated review would "
            "produce confident predictions about text the model never saw."
        ),
    )
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    encoder: str
    aspects: list[str]
    labels: list[str]
    max_length: int
    registry_version: str | None = Field(
        default=None,
        description="MLflow Model Registry version, if this artifact was promoted.",
    )
    run_id: str | None = None
    git_commit: str | None = None
    trained_at: str | None = None
    hyperparameters: dict = {}
    validation_metrics: dict = {}
