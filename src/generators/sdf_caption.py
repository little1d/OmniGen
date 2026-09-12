import hashlib
import json
from typing import Any

from ..data_loaders import DataSample
from ..sdf_caption import build_messages, parse_response, validate_response
from .base import BaseGenerator, register_generator


@register_generator("sdf_caption")
class SDFCaptionGenerator(BaseGenerator):
    """Generate grounded captions from precomputed molecular facts."""

    def generate_single(self, sample: DataSample) -> dict[str, Any] | None:
        if not self.validate_input(sample):
            return None
        if self.model_client is None:
            raise ValueError("Model client is required for SDF-caption generation")

        facts = sample.metadata["facts"]
        sdf = sample.metadata["sdf"]
        smiles = sample.metadata["smiles"]
        messages = build_messages(smiles, facts, sdf)
        max_out_len = int(self.config.get("max_out_len", 1536))
        if hasattr(self.model_client, "generate_with_metadata"):
            generation = self.model_client.generate_with_metadata(messages, max_out_len=max_out_len)
            raw_response = generation["content"]
        else:
            raw_response = self.model_client.generate(messages, max_out_len=max_out_len)
            generation = {
                "model": getattr(self.model_client, "model_name", None),
                "usage": {},
                "finish_reason": None,
                "log_id": None,
            }

        response = parse_response(raw_response)
        validation = validate_response(response, facts)
        if self.config.get("reject_invalid", True) and not validation["passed"]:
            raise ValueError("; ".join(validation["errors"]))

        prompt = "\n\n".join(
            f"[{message['role']}]\n{message['content']}" for message in messages
        )
        facts_json = json.dumps(facts, sort_keys=True, ensure_ascii=False)
        return {
            "id": sample.id,
            "molecule_id": sample.id,
            "display_key": sample.id,
            "source": "sdf_caption",
            "source_label": "SDF Caption",
            "formula": facts.get("composition", {}).get("formula"),
            "smiles": smiles,
            "caption": response["caption"],
            "sdf": sdf,
            "facts": facts,
            "prompt": prompt,
            "model": generation.get("model") or getattr(self.model_client, "model_name", None),
            "usage": generation.get("usage") or {},
            "generation": {
                "finish_reason": generation.get("finish_reason"),
                "log_id": generation.get("log_id"),
            },
            "validation": validation,
            "provenance": {
                "sdf_file": sample.metadata["sdf_filename"],
                "facts_file": sample.metadata["facts_filename"],
                "sdf_sha256": hashlib.sha256(sdf.encode("utf-8")).hexdigest(),
                "facts_sha256": hashlib.sha256(facts_json.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            },
        }

    def validate_input(self, sample: DataSample) -> bool:
        return all(
            key in sample.metadata
            for key in ("facts", "sdf", "smiles", "sdf_filename", "facts_filename")
        )

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": [
                "id",
                "caption",
                "smiles",
                "sdf",
                "facts",
                "validation",
            ],
        }
