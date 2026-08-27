"""API contract tests, driven through a fake service.

No model is loaded. That is deliberate: a test that reads a 265 MB checkpoint to
check that empty input returns 422 is testing the wrong layer, takes seconds instead
of milliseconds, and cannot run in CI.

The fake is installed by patching attributes on the module-level `service` object
that api.main already holds a reference to, so the routes are exercised exactly as
written — no dependency-injection scaffolding added purely for tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

FAKE_METADATA = {
    "encoder": "distilbert-base-uncased",
    "aspects": ["food", "service", "ambiance", "price", "misc"],
    "labels": ["absent", "negative", "neutral", "positive"],
    "max_length": 128,
    "registry_version": "4",
    "run_id": "deadbeef",
    "git_commit": "abc123",
    "trained_at": "2026-08-25T16:08:20+00:00",
    "hyperparameters": {"epochs": 4, "class_weights": "sqrt-inverse"},
    "validation_metrics": {"macro_f1": 0.598},
}

ASPECT_ORDER = ("food", "service", "ambiance", "price", "misc")


def fake_predict(text: str, explain: bool = False):
    """Canned prediction: food positive, everything else absent."""
    aspects = []
    for aspect in ASPECT_ORDER:
        mentioned = aspect == "food"
        words = None
        if explain and mentioned:
            words = [{"word": "great", "importance": 1.0},
                     {"word": "food", "importance": 0.4}]
        aspects.append(
            {
                "aspect": aspect,
                "label": "positive" if mentioned else "absent",
                "confidence": 0.71 if mentioned else 0.95,
                "mentioned": mentioned,
                "words": words,
            }
        )
    return aspects, len(text) > 600, 12.34


@pytest.fixture
def loaded_service(monkeypatch):
    from api import service as service_module

    svc = service_module.service
    monkeypatch.setattr(svc, "load", lambda: None)
    monkeypatch.setattr(svc, "model", object())      # makes is_loaded True
    monkeypatch.setattr(svc, "metadata", FAKE_METADATA)
    monkeypatch.setattr(svc, "predict", fake_predict)
    return svc


@pytest.fixture
def client(loaded_service):
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unloaded_client(monkeypatch):
    """A server that started but has no model — the state /health exists to catch."""
    from api import service as service_module

    svc = service_module.service
    monkeypatch.setattr(svc, "load", lambda: None)
    monkeypatch.setattr(svc, "model", None)
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_reports_ok_when_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}

    def test_reports_degraded_without_a_model(self, unloaded_client):
        """A health check that only proves the web server answered is decoration."""
        body = unloaded_client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False


class TestModelInfo:
    def test_returns_provenance(self, client):
        body = client.get("/model-info").json()
        assert body["registry_version"] == "4"
        assert body["run_id"] == "deadbeef"
        assert body["git_commit"] == "abc123"

    def test_aspect_order_is_preserved(self, client):
        """Label vectors are positional — index 0 means food in the data, the model
        and the API alike. Reordering here would silently mislabel everything."""
        assert client.get("/model-info").json()["aspects"] == list(ASPECT_ORDER)

    def test_503_when_no_model(self, unloaded_client):
        assert unloaded_client.get("/model-info").status_code == 503


class TestPredict:
    def test_happy_path(self, client):
        assert client.post("/predict", json={"text": "Great food."}).status_code == 200

    def test_always_returns_all_five_aspects(self, client):
        """Including the absent ones. Omitting them would make the response shape
        depend on the input, so the UI could not tell 'not discussed' from 'the API
        changed'."""
        body = client.post("/predict", json={"text": "Great food."}).json()
        assert [a["aspect"] for a in body["aspects"]] == list(ASPECT_ORDER)

    def test_mentioned_is_false_exactly_when_absent(self, client):
        body = client.post("/predict", json={"text": "Great food."}).json()
        for aspect in body["aspects"]:
            assert aspect["mentioned"] == (aspect["label"] != "absent")

    def test_no_words_without_explain(self, client):
        body = client.post("/predict", json={"text": "Great food."}).json()
        assert body["explained"] is False
        assert all(a["words"] is None for a in body["aspects"])

    def test_words_only_for_mentioned_aspects(self, client):
        """Explanations cost a backward pass each, so absent aspects do not get one."""
        body = client.post("/predict", json={"text": "Great food.", "explain": True}).json()
        assert body["explained"] is True
        for aspect in body["aspects"]:
            assert (aspect["words"] is not None) == aspect["mentioned"]

    def test_importance_is_within_range(self, client):
        body = client.post("/predict", json={"text": "Great food.", "explain": True}).json()
        words = next(a["words"] for a in body["aspects"] if a["mentioned"])
        assert all(0.0 <= w["importance"] <= 1.0 for w in words)

    def test_truncation_is_surfaced(self, client):
        """A silently truncated review would yield confident predictions about text
        the model never read."""
        short = client.post("/predict", json={"text": "Great food."}).json()
        long_body = client.post("/predict", json={"text": "word " * 200}).json()
        assert short["truncated"] is False
        assert long_body["truncated"] is True

    def test_response_echoes_stripped_text(self, client):
        body = client.post("/predict", json={"text": "  Great food.  "}).json()
        assert body["text"] == "Great food."

    def test_503_when_no_model(self, unloaded_client):
        response = unloaded_client.post("/predict", json={"text": "Great food."})
        assert response.status_code == 503


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"text": ""}, id="empty"),
            pytest.param({}, id="missing-field"),
            pytest.param({"text": 123}, id="wrong-type"),
            pytest.param({"text": "a" * 5001}, id="too-long"),
        ],
    )
    def test_rejected_with_422(self, client, payload):
        assert client.post("/predict", json=payload).status_code == 422

    def test_whitespace_only_is_rejected(self, client):
        """Field(min_length=1) accepts three spaces — length 3. The handler strips
        and re-checks, because schema validation and semantic validation are not
        the same job."""
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422
        assert "non-whitespace" in response.json()["detail"]

    def test_wrong_method(self, client):
        assert client.get("/predict").status_code == 405

    def test_unknown_route(self, client):
        assert client.get("/nope").status_code == 404

    def test_errors_do_not_leak_internals(self, client):
        body = client.post("/predict", json={"text": "   "}).text
        assert "Traceback" not in body
        assert "ml/src" not in body


class TestCORS:
    def test_allowed_origin_gets_the_header(self, client):
        """The React dev server depends on this."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_unlisted_origin_gets_no_allow_header(self, client):
        """Explicit origins rather than '*' — a wildcard works today and becomes a
        finding in any security review."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers


class TestOpenAPI:
    def test_schema_covers_every_route(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/predict", "/health", "/model-info"} <= set(paths)

    def test_docs_page_serves(self, client):
        """The interactive console TESTING.md tells you to use instead of curl."""
        assert client.get("/docs").status_code == 200
