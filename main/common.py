"""
Common utilities and argument handling for OmniGen unified entry points.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from src.config import OmniGenConfig


# Default configuration files for each data type
DEFAULT_CONFIGS = {
    "benchmark": "src/config/benchmark.yaml",
    "image_pair": "src/config/image_pair.yaml",
    "qa_pair": "src/config/qa_pair.yaml",
    "cot": "src/config/cot.yaml",
    "sdf_caption": "src/config/sdf_caption.yaml",
}

# Supported data types
SUPPORTED_DATA_TYPES = list(DEFAULT_CONFIGS.keys())


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for generation commands."""
    parser.add_argument(
        "--type", type=str, required=True, choices=SUPPORTED_DATA_TYPES, help="Type of data to generate"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML/JSON file (auto-selected if not provided)")
    parser.add_argument("--data-source", type=str, help="Override data source (Hugging Face dataset ID or local path)")
    parser.add_argument("--samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--output", type=str, help="Override output file path")
    parser.add_argument("--max-data-samples", type=int, help="Limit number of data samples to load")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")


def add_common_validation_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for validation commands."""
    parser.add_argument(
        "--type", type=str, required=True, choices=SUPPORTED_DATA_TYPES, help="Type of data to validate"
    )
    parser.add_argument("--file", type=str, required=True, help="Path to data file to validate")
    parser.add_argument("--config", type=str, help="Path to config file (auto-selected if not provided)")
    parser.add_argument("--output-dir", type=str, help="Override output directory for validation reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")


def add_type_specific_generation_args(parser: argparse.ArgumentParser, data_type: str) -> None:
    """Add type-specific arguments for generation."""
    if data_type == "benchmark":
        parser.add_argument("--question-type", type=str, help="Question type to generate (e.g. type_cls)")
    elif data_type == "image_pair":
        parser.add_argument(
            "--description-type",
            type=str,
            choices=["basic_description", "detailed_analysis"],
            help="Type of description to generate",
        )
    elif data_type == "qa_pair":
        parser.add_argument(
            "--qa-type",
            type=str,
            choices=["single_step", "multi_step"],
            help="Type of QA conversation to generate",
        )
    elif data_type == "cot":
        parser.add_argument("--cot-type", type=str, choices=["text_only", "multimodal"], help="Type of CoT to generate")
        parser.add_argument(
            "--stages",
            type=str,
            nargs="+",
            choices=["draft", "reason"],
            help="Stages to run: draft (question/solution), reason (thinking trajectories)",
        )
        parser.add_argument(
            "--input-json",
            type=str,
            help="Input JSON file for incremental generation (typically used with --stages reason)",
        )


def load_config_for_type(data_type: str, config_path: Optional[str] = None) -> OmniGenConfig:
    """Load configuration for the specified data type."""
    if config_path is None:
        config_path = DEFAULT_CONFIGS.get(data_type)
        if config_path is None:
            raise ValueError(f"No default config available for data type: {data_type}")

    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        config = OmniGenConfig.from_file(config_path)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def apply_generation_overrides(config: OmniGenConfig, args: argparse.Namespace, data_type: str) -> None:
    """Apply command-line argument overrides to configuration."""
    # Common overrides
    if args.data_source:
        config.dataset_path = args.data_source

    if args.samples is not None:
        config.num_samples = args.samples

    if args.output:
        config.output_path = args.output

    if args.verbose:
        config.verbose = True

    if args.max_data_samples:
        if "loader_kwargs" not in config.generator_config:
            config.generator_config["loader_kwargs"] = {}
        config.generator_config["loader_kwargs"]["max_samples"] = args.max_data_samples

    # Type-specific overrides
    if data_type == "benchmark" and hasattr(args, 'question_type') and args.question_type:
        config.generator_config["question_types"] = [args.question_type]

    elif data_type == "image_pair" and hasattr(args, 'description_type') and args.description_type:
        config.generator_config["description_types"] = [args.description_type]

    elif data_type == "qa_pair" and hasattr(args, 'qa_type') and args.qa_type:
        config.generator_config["qa_types"] = [args.qa_type]

    elif data_type == "cot":
        if hasattr(args, 'cot_type') and args.cot_type:
            config.generator_config["cot_types"] = [args.cot_type]
        if hasattr(args, 'stages') and args.stages:
            config.generator_config["stages"] = args.stages


def apply_validation_overrides(config: OmniGenConfig, args: argparse.Namespace) -> None:
    """Apply validation-specific command-line argument overrides."""
    if args.output_dir:
        config.validation_config.output_dir = args.output_dir

    if args.verbose:
        config.validation_config.verbose = True


def print_config_summary(config: OmniGenConfig, data_type: str, mode: str = "generation") -> None:
    """Print a summary of the current configuration."""
    print(f"\n📋 {mode.title()} Configuration Summary:")
    print(f"  - Data type: {data_type}")

    if mode == "generation":
        print(f"  - Data source: {config.dataset_path}")
        print(f"  - Generator type: {config.generator_type}")
        print(f"  - Target samples: {config.num_samples}")
        print(f"  - Output path: {config.output_path}")
        print(f"  - Verbose: {config.verbose}")
    elif mode == "validation":
        print(f"  - Validation output: {config.validation_config.output_dir}")
        print(f"  - Verbose: {config.validation_config.verbose}")
        print(f"  - Overall threshold: {config.validation_config.overall_threshold}")


def handle_error(error: Exception, context: str = "") -> None:
    """Handle and format errors consistently."""
    error_msg = f"❌ Error"
    if context:
        error_msg += f" in {context}"
    error_msg += f": {error}"

    print(error_msg, file=sys.stderr)

    # Print additional help for common errors
    if isinstance(error, FileNotFoundError):
        print("💡 Tip: Check that the file path is correct and the file exists.", file=sys.stderr)
    elif isinstance(error, ValueError):
        print("💡 Tip: Check that the data type and arguments are valid.", file=sys.stderr)
    elif isinstance(error, RuntimeError):
        print("💡 Tip: Check the configuration file format and model settings.", file=sys.stderr)


def get_default_output_path(data_type: str, mode: str = "generation") -> str:
    """Get default output path for a data type and mode."""
    if mode == "generation":
        return f"output/{data_type}.json"
    elif mode == "validation":
        return f"output/{data_type}_validation"
    else:
        return f"output/{data_type}_{mode}"


def validate_data_type(data_type: str) -> None:
    """Validate that the data type is supported."""
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(f"Unsupported data type: {data_type}. " f"Supported types: {', '.join(SUPPORTED_DATA_TYPES)}")


def validate_file_exists(file_path: str, description: str = "file") -> None:
    """Validate that a file exists."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The {description} does not exist: {file_path}")


def get_config_info(data_type: str) -> Dict[str, str]:
    """Get configuration information for a data type."""
    return {
        "data_type": data_type,
        "default_config": DEFAULT_CONFIGS.get(data_type, "Not available"),
        "supported": data_type in SUPPORTED_DATA_TYPES,
    }
