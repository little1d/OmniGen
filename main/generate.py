"""
Unified generation entry point for OmniGen.

Usage:
    # Basic generation
    python -m main.generate --type benchmark --config src/config/benchmark.yaml --samples 10
    python -m main.generate --type image_pair --data-source "SpectrumWorld/multimodal-spectroscopic-seed-datasets-1000" --samples 5
    python -m main.generate --type cot --config src/config/cot.yaml --output cot_data.json
    python -m main.generate --type qa_pair --config src/config/qa_pair.yaml --output qa_pairs.json
    python -m main.generate --type sdf_caption --data-source data/sdf_caption

    # CoT staged generation
    python -m main.generate --type cot --stages draft --cot-type text_only --samples 500
    python -m main.generate --type cot --stages draft --cot-type multimodal --samples 500

    # CoT incremental generation (add reasoning to existing draft results)
    python -m main.generate --type cot --input-json output/cot_text_only_500.json \\
        --stages reason --cot-type text_only --output output/cot_text_only_with_reasoning.json
    python -m main.generate --type cot --input-json output/cot_multimodal_500.json \\
        --stages reason --cot-type multimodal --output output/cot_multimodal_with_reasoning.json
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import OmniGenConfig
from src.core import OmniGenEngine
from .common import (
    add_common_generation_args,
    add_type_specific_generation_args,
    load_config_for_type,
    apply_generation_overrides,
    print_config_summary,
    handle_error,
    validate_data_type,
)


async def run_generation(config: OmniGenConfig, data_type: str, input_json: Optional[str] = None) -> bool:
    """Run the generation process."""
    try:
        # Create engine
        engine = OmniGenEngine(config)

        # Load dataset
        if input_json:
            print(f"📂 Loading existing data from: {input_json}")
            engine.load_from_json(input_json)
            use_sequential = True  # Use sequential processing for incremental generation
        else:
            engine.load_dataset()
            use_sequential = data_type == "sdf_caption"

        # Run generation
        print(f"\n🚀 Starting {data_type} generation...")
        results = await engine.run(use_sequential=use_sequential)

        if results:
            print(f"\n✅ Generation completed successfully!")
            print(f"   Generated {len(results)} samples")
            print(f"   Results saved to: {config.output_path}")

            # Show sample output if verbose
            if config.verbose and results:
                print(f"\n📋 Sample output:")
                sample = results[0]
                print(f"   ID: {sample.get('id', 'N/A')}")
                if data_type == "benchmark":
                    print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
                    print(f"   Answer: {sample.get('answer', 'N/A')}")
                elif data_type == "image_pair":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    print(f"   Image: {sample.get('image', 'N/A')}")
                    print(f"   Text: {sample.get('text', 'N/A')[:100]}...")
                elif data_type == "qa_pair":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    print(f"   Image: {sample.get('image', 'N/A')}")
                    conversations = sample.get('conversations', [])
                    print(f"   Conversations: {len(conversations)}")
                    if conversations:
                        print(f"   First Q: {conversations[0].get('value', 'N/A')[:100]}...")
                        if len(conversations) > 1:
                            print(f"   First A: {conversations[1].get('value', 'N/A')[:100]}...")
                elif data_type == "cot":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    if sample.get('type') == "cot multimodal":
                        print(f"   Image: {sample.get('image', 'N/A')}")
                    print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
                    print(f"   Solution: {sample.get('solution', 'N/A')[:100]}...")
                    if 'claude_thinking_trajectories' in sample:
                        print(f"   Claude reasoning: {sample.get('claude_thinking_trajectories', 'N/A')[:100]}...")
                    if 'interns1_thinking_trajectories' in sample:
                        print(f"   InternS1 reasoning: {sample.get('interns1_thinking_trajectories', 'N/A')[:100]}...")
                elif data_type == "sdf_caption":
                    print(f"   Formula: {sample.get('formula', 'N/A')}")
                    print(f"   Caption: {sample.get('caption', 'N/A')[:200]}...")
                else:
                    print(f"   Content: {str(sample)[:200]}...")

            return True
        else:
            print(f"\n❌ Generation failed - no valid samples generated")
            return False

    except Exception as e:
        handle_error(e, "generation")
        return False


def main():
    """Main entry point for unified generation."""
    parser = argparse.ArgumentParser(
        description="OmniGen Unified Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m main.generate --type benchmark --samples 10
  python -m main.generate --type image_pair --data-source "SpectrumWorld/molpuzzle-seed-datasets" --samples 5
  python -m main.generate --type benchmark --config custom_config.yaml --output my_benchmark.json
  python -m main.generate --type sdf_caption --data-source data/sdf_caption
        """,
    )

    # Add common arguments
    add_common_generation_args(parser)

    # Parse arguments to get data type, then add type-specific arguments
    args, remaining = parser.parse_known_args()

    try:
        validate_data_type(args.type)

        # Add type-specific arguments and re-parse
        add_type_specific_generation_args(parser, args.type)
        args = parser.parse_args()

        # Load configuration
        config = load_config_for_type(args.type, args.config)

        # Apply overrides
        apply_generation_overrides(config, args, args.type)

        # Print configuration summary
        print_config_summary(config, args.type, "generation")

        # Get input_json if provided (for CoT incremental generation)
        input_json = getattr(args, 'input_json', None) if hasattr(args, 'input_json') else None

        # Run generation
        success = asyncio.run(run_generation(config, args.type, input_json))

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        handle_error(e, "argument parsing or configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
