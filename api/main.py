"""FastAPI application: /health, /model-info, /predict.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs

The model is loaded once during startup via the lifespan handler, not on the
first request. That means the process either starts with a working model or
fails loudly at boot — which is what you want from a container: a crash-looping
pod is obvious, whereas a pod that accepts traffic and returns 500s looks
healthy to everything except its users.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from api.service import service

logger = logging.getLogger("absa.api")

# Explicit origins, not "*". A wildcard would work today and become a finding in
# any security review, so the list is stated even though this API holds no
# secrets.
#
# Overridable via ALLOWED_ORIGINS (comma-separated) because the correct value is
# a deployment fact, not a code fact. The defaults cover the Vite dev server
# (5173) and the containerised page nginx serves (8080) — that second one was
# missing at first, and the failure mode is worth knowing: the page loads
# perfectly and every request fails, visible only in the browser console. curl
# does not enforce CORS, so nothing in a terminal test would have caught it.
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading model...")
    service.load()
    info = service.info()
    logger.info(
        "model ready: %s (registry version %s)",
        info["encoder"],
        info["registry_version"],
    )
    yield
    # Nothing to release: torch frees the weights when the process exits, and
    # there are no connections or file handles held open.
    logger.info("shutting down")


app = FastAPI(
    title="Aspect-Based Sentiment Analysis",
    description=(
        "Predicts sentiment for five restaurant-review aspects "
        "(food, service, ambience, price, misc), with optional per-word "
        "attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare host to the interactive docs.

    Without this, clicking the URL uvicorn prints in the terminal lands on
    {"detail":"Not Found"} — correct, since no resource exists at /, but it reads
    like a broken server. The useful thing to serve at the root of an API with no
    root resource is the console that lists the resources it does have.

    Excluded from the OpenAPI schema: it is a convenience for humans, not part of
    the contract the frontend codes against.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe.

    Deliberately reports whether the MODEL is loaded, not merely whether the web
    server answered. A process that is up but has no model is not healthy, and a
    health check that cannot tell the difference is decoration.
    """
    loaded = service.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """What model is running, where it came from, and how well it scored.

    This is the answer to "which model is in production?" — registry version,
    MLflow run id, and the git commit the code was at. Without it, a served
    artifact is an anonymous 265 MB file.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )
    return ModelInfoResponse(**service.info())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for all five aspects of one review.

    Always returns all five aspects, including the ones scored 'absent'. The
    alternative — omitting them — would make the UI guess whether a missing
    aspect meant "not discussed" or "the API changed", and it makes the response
    shape depend on the input, which is unpleasant to consume.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )

    text = request.text.strip()
    if not text:
        # Pydantic's min_length=1 accepts "   ", which is not a review. Checked
        # after stripping so whitespace-only input is a clean 422 rather than a
        # confident prediction about nothing.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="text must contain non-whitespace characters",
        )

    aspects, truncated, latency_ms = service.predict(text, explain=request.explain)

    return PredictResponse(
        text=text,
        aspects=aspects,
        explained=request.explain,
        truncated=truncated,
        latency_ms=latency_ms,
    )
