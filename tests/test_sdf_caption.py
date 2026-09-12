import asyncio
import json

import httpx

from src.config import OmniGenConfig, ValidatorConfig
from src.core.generation_engine import OmniGenEngine
from src.core.validator_engine import ValidatorEngine
from src.data_loaders import create_loader_for_source
from src.generators.sdf_caption import SDFCaptionGenerator
from src.sdf_caption import build_messages, validate_response
from src.validators.sdf_caption import SDFCaptionValidator


FACTS = {
    "molecule_id": "mol-1",
    "composition": {"formula": "C2H6O", "formal_charge": 0},
    "topology": {"ring_count": 0, "ring_sizes": []},
    "geometry": {
        "radius_of_gyration_angstrom": 1.2345,
        "asphericity": 0.1234,
        "shape_class": "rod_like",
    },
}
SDF = """mol-1
  RDKit          3D

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
$$$$
"""
CAPTION = "该分子为分子式C2H6O的中性含氧小分子。当前构象呈现连续的含碳骨架与局部极性区域。"


class FakeModel:
    model_name = "fake-model"

    def generate_with_metadata(self, messages, max_out_len):
        self.messages = messages
        self.max_out_len = max_out_len
        return {
            "content": json.dumps({"caption": CAPTION}, ensure_ascii=False),
            "model": self.model_name,
            "usage": {"completion_tokens": 20},
            "finish_reason": "stop",
            "log_id": "test-log",
        }


def test_prompt_contains_smiles_facts_and_sdf():
    messages = build_messages("CCO", FACTS, SDF)
    user_prompt = messages[1]["content"]

    assert "CCO" in user_prompt
    assert '"radius_of_gyration_angstrom": 1.2345' in user_prompt
    assert SDF.strip() in user_prompt
    assert '只输出 JSON 对象：{"caption":"..."}' in user_prompt
    assert "reasoning_trace" not in user_prompt
    assert "该分子是一种" not in user_prompt
    assert "总体而言" not in user_prompt
    assert "不是可拼接的模板" in user_prompt


def test_response_validation_is_flexible_but_grounded():
    assert validate_response({"caption": CAPTION}, FACTS)["passed"]
    assert validate_response({"caption": CAPTION, "reasoning": []}, FACTS)["passed"] is False
    assert validate_response(
        {"caption": "该分子的分子式为C2H6O，回转半径为9.9999。"},
        FACTS,
    )["passed"] is False


def test_loader_generator_and_validator_work_together(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "mol-1.sdf").write_text(SDF, encoding="utf-8")
    (input_dir / "mol-1.facts.json").write_text(
        json.dumps(FACTS, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "reference.json").write_text(
        json.dumps([{"molecule_id": "mol-1", "smiles": "CCO"}]),
        encoding="utf-8",
    )

    loader = create_loader_for_source(input_dir)
    dataset = loader.load(input_dir)
    model = FakeModel()
    output = SDFCaptionGenerator(
        {"max_out_len": 512, "reject_invalid": True},
        model,
    ).generate_single(dataset[0])

    assert output["caption"] == CAPTION
    assert output["smiles"] == "CCO"
    assert output["sdf"] == SDF
    assert output["validation"]["passed"]
    assert model.max_out_len == 512

    validator = SDFCaptionValidator(None, ValidatorConfig())
    result = asyncio.run(validator.validate_sample(output))
    assert result.is_valid
    assert result.overall_score == 10.0

    output_path = tmp_path / "captions.jsonl"
    output_path.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    config = OmniGenConfig(
        model_type="mock",
        validation_config=ValidatorConfig(
            output_dir=str(tmp_path / "validation"),
            verbose=False,
        ),
    )
    report = asyncio.run(
        ValidatorEngine(config).validate_file(str(output_path), "sdf_caption")
    )
    assert report["statistics"]["valid_samples"] == 1
    assert report["validation_results"][0]["sample_id"] == "mol-1"


def test_sdf_caption_config_builds_validator_config():
    config = OmniGenConfig.from_file("src/config/sdf_caption.yaml")
    assert config.generator_type == "sdf_caption"
    assert config.num_samples == 0
    assert isinstance(config.validation_config, ValidatorConfig)


def test_generation_engine_writes_jsonl(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "mol-1.sdf").write_text(SDF, encoding="utf-8")
    facts = {**FACTS, "smiles": "CCO"}
    (input_dir / "mol-1.facts.json").write_text(
        json.dumps(facts, ensure_ascii=False),
        encoding="utf-8",
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps({"caption": CAPTION})},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    output_path = tmp_path / "captions.jsonl"
    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OmniGenConfig(
        dataset_path=str(input_dir),
        generator_type="sdf_caption",
        generator_config={
            "max_out_len": 512,
            "reject_invalid": True,
            "overwrite_output": True,
        },
        model_type="modelhub",
        model_config={
            "model_name": "fake-model",
            "ak": "test-ak",
            "endpoint": "https://example.test/crawl",
            "client": client,
        },
        num_samples=0,
        output_path=str(output_path),
        verbose=False,
    )

    rows = asyncio.run(OmniGenEngine(config).run(use_sequential=True))
    client.close()

    saved = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == len(saved) == 1
    assert saved[0]["id"] == "mol-1"
