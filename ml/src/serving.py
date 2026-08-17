"""The MLflow pyfunc wrapper — the form the model takes once it leaves training.

This file exists separately from tracking.py for a reason worth understanding,
because it is a mistake that only shows up at load time.

MLflow serialises a PythonModel with cloudpickle, which stores the *class by
reference* — module name plus qualified name — not the class body. Whatever
module this class is defined in must therefore be importable when the model is
loaded back. MLflow makes that possible via ``code_paths``, which copies listed
source files into the artifact and puts them on sys.path.

Defining the wrapper inside tracking.py meant the pickle referenced a module
called ``tracking``, which was not in code_paths, and loading failed with
ModuleNotFoundError. The alternative — shipping tracking.py too — would drag
matplotlib, sklearn and the whole experiment-logging apparatus into every
serving container to support a class that needs none of it.

So the split is along the real seam: this module is what ships *inside* the
artifact, tracking.py is build-time only and stays behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import torch


class ABSAModel(mlflow.pyfunc.PythonModel):
    """Packages weights + tokenizer + metadata behind one predict call.

    Why wrap at all, rather than logging the artifact folder directly: as of
    MLflow 3.x, register_model refuses anything that is not a properly logged
    model — a bare directory fails with "Unable to find a logged_model". The
    Model Registry is what turns "some files on my laptop" into
    ``models:/absa-distilbert/3``, so the wrapper is the price of version numbers.

    It also forces the artifact to be self-contained. Weights alone are not a
    model: load them against a different tokenizer vocabulary and the result is
    silent nonsense rather than an error.
    """

    def load_context(self, context: Any) -> None:
        # Imported inside the method, not at module scope. At load time this
        # class is reconstructed inside MLflow's environment, where model.py is
        # only importable after MLflow has put its code/ directory on sys.path —
        # which happens after this module is first imported.
        from transformers import AutoTokenizer

        from model import AspectSentimentModel

        model_dir = Path(context.artifacts["model_dir"])
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = AspectSentimentModel(encoder_name=self.metadata["encoder"])
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def predict(
        self, context: Any, model_input: list[str], params: Any = None
    ) -> list[dict]:
        """Annotated ``list[str]`` deliberately: MLflow 3 infers the model's
        input signature from this type hint and records it in the registry, so
        the artifact documents its own interface. An un-annotated ``Any`` logs
        the model with no signature at all and warns while doing it.
        """
        from model import predict as run_prediction

        # MLflow hands DataFrames in from the REST path and lists in from direct
        # Python calls; accept both rather than making callers care.
        if hasattr(model_input, "iloc"):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = [str(text) for text in model_input]

        return run_prediction(self.model, texts, self.tokenizer, "cpu")
