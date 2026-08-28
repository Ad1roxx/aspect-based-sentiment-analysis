"""Tests that need the real 265 MB artifact. Skipped when it is absent.

These automate the three things verified by hand in earlier sprints:

  * the artifact reloads from disk and predicts (Sprint 1)
  * explanations genuinely differ per aspect, and occluding the top word changes
    the prediction (Sprint 3)
  * the registry round-trip actually loads, which is the bug Sprint 2 shipped and
    then caught (Sprint 2)

They are marked ``integration`` and skipped when ml/models/ is empty, so CI stays
green without a GPU while a developer with a trained model gets full coverage.

Run just these:      pytest -m integration
Run everything else: pytest -m "not integration"
"""

from __future__ import annotations

import pytest
import torch

from conftest import ARTIFACT_DIR, artifact_required

pytestmark = [pytest.mark.integration, artifact_required]


@pytest.fixture(scope="module")
def loaded():
    """Load the artifact once for the whole module — it is 265 MB."""
    from predict import load_model

    device = torch.device("cpu")
    model, tokenizer, metadata = load_model(device=device)
    return model, tokenizer, metadata, device


class TestArtifact:
    def test_required_files_are_present(self):
        """A model without its tokenizer is not a model: load the weights against a
        different vocabulary and you get silent nonsense rather than an error."""
        for name in ("model.pt", "metadata.json", "tokenizer.json", "tokenizer_config.json"):
            assert (ARTIFACT_DIR / name).is_file(), f"missing {name}"

    def test_metadata_carries_provenance(self, loaded):
        """The API serves this directory rather than a registry URI, so the artifact
        has to be able to say what it is. Without these, 'which model is in
        production?' is unanswerable from the running service."""
        _, _, metadata, _ = loaded
        for field in ("run_id", "git_commit", "trained_at", "registry_version"):
            assert metadata.get(field), f"{field} missing or empty"

    def test_aspect_order_matches_the_code(self, loaded):
        """Label vectors are positional. If a retrained artifact ever disagreed with
        ASPECTS, every prediction would be silently mislabelled."""
        from data import ASPECTS

        _, _, metadata, _ = loaded
        assert tuple(metadata["aspects"]) == ASPECTS


class TestPrediction:
    def test_returns_one_entry_per_aspect(self, loaded):
        from data import ASPECTS, LABEL_NAMES
        from model import predict

        model, tokenizer, _, device = loaded
        result = predict(model, ["The pasta was incredible."], tokenizer, device)[0]

        assert set(result) == set(ASPECTS)
        for entry in result.values():
            assert entry["label"] in LABEL_NAMES
            assert 0.0 <= entry["confidence"] <= 1.0

    @pytest.mark.parametrize(
        "text,aspect,expected",
        [
            ("The pasta was incredible.", "food", "positive"),
            ("It is way too expensive for what you get.", "price", "negative"),
            ("I walked past it on my way to work.", "food", "absent"),
        ],
    )
    def test_known_examples(self, loaded, text, aspect, expected):
        """Spot-checks, not a metric. Deliberately excludes service, which the model
        is known to miss — see TESTING.md section 6."""
        from model import predict

        model, tokenizer, _, device = loaded
        assert predict(model, [text], tokenizer, device)[0][aspect]["label"] == expected

    def test_batching_matches_single_prediction(self, loaded):
        """Dynamic padding means a batched sentence is padded differently from a
        lone one. attention_mask should make that irrelevant — if it does not,
        results would depend on what else was in the request."""
        from model import predict

        model, tokenizer, _, device = loaded
        texts = ["Great food.", "The waiter was extremely rude to us all evening."]

        batched = predict(model, texts, tokenizer, device)
        alone = [predict(model, [t], tokenizer, device)[0] for t in texts]

        for batch_result, single_result in zip(batched, alone):
            for aspect in batch_result:
                assert batch_result[aspect]["label"] == single_result[aspect]["label"]


class TestExplanations:
    def test_importances_are_normalised(self, loaded):
        from explain import explain

        model, tokenizer, _, device = loaded
        result = explain(model, "The pasta was incredible.", tokenizer, device, "food")

        scores = [score for _, score in result["words"]]
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert max(scores) == pytest.approx(1.0)

    def test_no_wordpiece_fragments_or_special_tokens_leak(self, loaded):
        from explain import explain

        model, tokenizer, _, device = loaded
        result = explain(model, "It was overpriced.", tokenizer, device, "price")

        words = [word for word, _ in result["words"]]
        assert not any(word.startswith("##") for word in words)
        assert not any(word in ("[CLS]", "[SEP]") for word in words)

    def test_explanations_differ_between_aspects(self, loaded):
        """The reason attention was rejected. One [CLS] vector feeds all five heads,
        so [CLS] attention is identical for every aspect. Gradient x input takes the
        gradient of ONE aspect's logit, so it genuinely varies — and if this ever
        stopped being true, the UI would show five identical highlights.
        """
        from explain import explain

        model, tokenizer, _, device = loaded
        text = "The sushi was fresh and delicious but our waiter was incredibly rude."

        food = dict(explain(model, text, tokenizer, device, "food")["words"])
        service = dict(explain(model, text, tokenizer, device, "service")["words"])

        assert food.keys() == service.keys()
        assert any(
            food[word] != pytest.approx(service[word], abs=1e-4) for word in food
        ), "food and service produced identical attributions"

    def test_occluding_the_top_word_weakens_the_prediction(self, loaded):
        """Causal validation of the attribution ranking.

        Gradient x input is a first-order approximation, so its output needs
        checking against something causal. If masking the highest-attributed word
        does not change the prediction, the ranking is not finding what the model
        depends on.

        Two details this test got wrong the first time, both worth keeping:

        * The occlusion is CASE-INSENSITIVE. The tokenizer is uncased, so it
          returns "cosy" for a sentence containing "Cosy". A plain str.replace
          matched nothing, so the "occluded" text was identical to the original
          and the test was comparing a sentence with itself — passing or failing
          for reasons unrelated to attribution.
        * The assertion is a CONFIDENCE DROP, not a label flip. A label flip
          requires the masked word to be worth more than the model's entire
          margin, which a well-trained model often survives. Demanding a flip
          makes the test fail as the model gets better.
        """
        import re

        from explain import explain
        from model import predict

        model, tokenizer, _, device = loaded
        text = "Cosy little place, though it is a bit overpriced for what you get."

        result = explain(model, text, tokenizer, device, "price")
        top_word = max(result["words"], key=lambda pair: pair[1])[0]

        occluded = re.sub(re.escape(top_word), "[MASK]", text, flags=re.IGNORECASE)
        assert occluded != text, f"occlusion of {top_word!r} did not alter the text"

        after = predict(model, [occluded], tokenizer, device)[0]["price"]
        weakened = after["label"] != result["label"] or (
            after["confidence"] < result["confidence"]
        )
        assert weakened, (
            f"masking {top_word!r} left {result['label']} at "
            f"{after['confidence']:.3f} vs {result['confidence']:.3f}"
        )


class TestPerformanceFloor:
    def test_test_split_macro_f1_has_not_regressed(self, loaded):
        """A floor, not an exact match.

        Asserting exactly 0.6287 would fail the moment the model is *improved*,
        which is a test that punishes progress. The floor catches the failure that
        matters: an artifact that loads but is broken, or a retrain that quietly
        made things worse.
        """
        from torch.utils.data import DataLoader

        from data import load_splits
        from train import AspectDataset, evaluate, make_collate_fn

        model, tokenizer, _, device = loaded
        _, _, test_examples = load_splits()

        loader = DataLoader(
            AspectDataset(test_examples),
            batch_size=64,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer),
        )
        metrics, _, _ = evaluate(model, loader, device)
        assert metrics["macro_f1"] > 0.55, f"macro-F1 fell to {metrics['macro_f1']:.4f}"


class TestRegistryRoundTrip:
    def test_registered_model_loads_and_predicts(self):
        """The bug Sprint 2 shipped: a model that registered cleanly and then failed
        to load with ModuleNotFoundError, because cloudpickle stores the wrapper
        class by module reference and tracking.py was not in code_paths.

        Registering an artifact is not the same as verifying it loads. This is the
        test that would have caught it.
        """
        import mlflow

        from tracking import REGISTERED_MODEL_NAME, TRACKING_URI

        mlflow.set_tracking_uri(TRACKING_URI)
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if not versions:
            pytest.skip("no registered model versions in the local registry")

        latest = max(versions, key=lambda v: int(v.version))
        model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
        )

        result = model.predict(["The pasta was incredible."])[0]
        assert result["food"]["label"] == "positive"


class TestRealModelService:
    """Exercises the actual service layer, not the fake.

    The API tests replace ModelService.predict with a fake so they can run without a
    checkpoint, and the other integration tests call model.predict directly. That
    left the glue between the two — truncation counting, the explain-only-if-
    mentioned rule, the response dict shape — covered by nothing at all. This is
    where a real bug would hide: both sides work, and the wiring does not.
    """

    @pytest.fixture(scope="class")
    def real_service(self):
        from api.service import ModelService

        service = ModelService()
        service.load()
        return service

    def test_loads(self, real_service):
        assert real_service.is_loaded

    def test_returns_all_five_aspects_in_order(self, real_service):
        from data import ASPECTS

        aspects, _, _ = real_service.predict("The pasta was incredible.")
        assert [a["aspect"] for a in aspects] == list(ASPECTS)

    def test_mentioned_matches_label(self, real_service):
        aspects, _, _ = real_service.predict("The pasta was incredible.")
        for aspect in aspects:
            assert aspect["mentioned"] == (aspect["label"] != "absent")

    def test_no_words_unless_explain_requested(self, real_service):
        aspects, _, _ = real_service.predict("The pasta was incredible.", explain=False)
        assert all(a["words"] is None for a in aspects)

    def test_explains_only_mentioned_aspects(self, real_service):
        """The cost rule: one backward pass per detected aspect, and none for the
        rest. If this inverted, a request would do five backward passes to explain
        why four topics were never discussed."""
        aspects, _, _ = real_service.predict("The pasta was incredible.", explain=True)
        for aspect in aspects:
            assert (aspect["words"] is not None) == aspect["mentioned"]

    def test_truncation_flag_tracks_the_token_limit(self, real_service):
        """Counted with special tokens, since [CLS] and [SEP] occupy two of the 128
        positions — an off-by-two here would mislabel borderline reviews."""
        assert real_service.is_truncated("Great food.") is False
        assert real_service.is_truncated("word " * 300) is True

    def test_latency_is_reported(self, real_service):
        _, _, latency_ms = real_service.predict("Great food.")
        assert latency_ms > 0

    def test_info_exposes_provenance(self, real_service):
        info = real_service.info()
        assert info["registry_version"]
        assert info["run_id"]
        assert info["git_commit"]
