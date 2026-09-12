from typing import Any

from .base import BaseModel
from .gemini_api import Gemini
from .intern_api import InternAPI, InternS1, InternVL
from .modelhub_api import ModelHub


MODEL_TYPES = {
    "gemini": Gemini,
    "interns1": InternS1,
    "internvl": InternVL,
    "modelhub": ModelHub,
}


def create_model(model_type: str, config: dict[str, Any]) -> BaseModel:
    try:
        model_class = MODEL_TYPES[model_type.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: {', '.join(sorted(MODEL_TYPES))}"
        ) from None
    return model_class(**config)


__all__ = [
    "BaseModel",
    "Gemini",
    "InternAPI",
    "InternS1",
    "InternVL",
    "ModelHub",
    "create_model",
]
