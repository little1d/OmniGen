import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import re

project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


def resolve_env_vars(obj):
    """Recursively resolve ${VAR} in dict/list/str using environment variables."""
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(i) for i in obj]
    elif isinstance(obj, str):

        def replacer(match):
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replacer, obj)
    else:
        return obj


@dataclass
class ModelConfig:
    # ModelHub API Configuration
    modelhub_ak: str | None = os.getenv("MODELHUB_AK")
    modelhub_endpoint: str | None = os.getenv("MODELHUB_ENDPOINT")
    modelhub_model_name: str | None = os.getenv("MODELHUB_MODEL_NAME")
    modelhub_log_id: str | None = os.getenv("MODELHUB_LOG_ID")

    # InternVL API Configuration
    internvl_api_key: str | None = os.getenv("INTERNVL_API_KEY")
    internvl_base_url: str | None = os.getenv("INTERNVL_BASE_URL")
    internvl_model_name: str | None = os.getenv("INTERNVL_MODEL_NAME")

    # InternS1 API Configuration
    interns1_api_key: str | None = os.getenv("INTERNS1_API_KEY")
    interns1_base_url: str | None = os.getenv("INTERNS1_BASE_URL")
    interns1_model_name: str | None = os.getenv("INTERNS1_MODEL_NAME")

    # Gemini API Configuration (via OpenRouter)
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    gemini_base_url: str | None = os.getenv("GEMINI_BASE_URL")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "google/gemini-2.5-pro")


@dataclass
class ValidatorConfig:
    """Validator configuration"""

    # General settings
    enabled: bool = True
    output_dir: str = "omnigenlog/validation_reports"
    verbose: bool = True

    # Threshold settings
    overall_threshold: float = 7.0
    critical_threshold: float = 5.0

    # Data type specific configs
    benchmark_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "question_quality_weight": 1.5,
            "answer_correctness_weight": 2.0,
            "choices_quality_weight": 1.0,
            "image_relevance_weight": 1.2,
            "strict_mode": False,
        }
    )

    cot_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "question_quality_weight": 1.0,
            "solution_quality_weight": 1.5,
            "reasoning_quality_weight": 1.0,
            "attempt_quality_weight": 1.0,
            "image_relevance_weight": 1.0,
            "min_score": 5.0,
        }
    )

    image_pair_config: Dict[str, Any] = field(default_factory=lambda: {})

    qa_pair_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "conversation_structure_weight": 1.0,
            "conversation_quality_weight": 1.5,
            "image_relevance_weight": 1.0,
            "min_conversations": 2,
            "max_conversations": 8,
        }
    )


@dataclass
class OmniGenConfig:
    # dataset config
    dataset_path: str = "data"

    # generator config
    generator_type: str = "benchmark"
    generator_config: Dict[str, Any] = field(default_factory=dict)

    # model config
    model_type: str = "mock"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # validation_config
    validation_config: ValidatorConfig = field(default_factory=ValidatorConfig)

    # other config
    num_samples: int = 10
    output_path: str = "output"
    verbose: bool = True
    max_retries: int = 3

    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        model_configs = ModelConfig()

        if model_type.lower() == "modelhub":
            return {
                "ak": model_configs.modelhub_ak,
                "endpoint": model_configs.modelhub_endpoint,
                "model_name": model_configs.modelhub_model_name,
                "log_id": model_configs.modelhub_log_id,
            }
        elif model_type.lower() == "internvl":
            return {
                "api_key": model_configs.internvl_api_key,
                "base_url": model_configs.internvl_base_url,
                "model_name": model_configs.internvl_model_name,
            }
        elif model_type.lower() == "interns1":
            return {
                "api_key": model_configs.interns1_api_key,
                "base_url": model_configs.interns1_base_url,
                "model_name": model_configs.interns1_model_name,
            }
        elif model_type.lower() == "gemini":
            return {
                "api_key": model_configs.gemini_api_key,
                "base_url": model_configs.gemini_base_url,
                "model_name": model_configs.gemini_model_name,
            }
        else:
            return {}

    @classmethod
    def from_file(cls, config_path: str) -> "OmniGenConfig":
        """load config from file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"unsupported config file format: {config_path.suffix}")

        # 递归替换环境变量
        data = resolve_env_vars(data)
        return cls(**data)

    def to_file(self, config_path: str) -> None:
        """save config to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_path": self.dataset_path,
            "generator_type": self.generator_type,
            "generator_config": self.generator_config,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "num_samples": self.num_samples,
            "output_path": self.output_path,
            "verbose": self.verbose,
            "max_retries": self.max_retries,
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, ensure_ascii=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"unsupported config file format: {config_path.suffix}")
