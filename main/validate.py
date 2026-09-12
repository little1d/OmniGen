"""
Unified validation entry point for OmniGen.

Usage:
    python -m main.validate --type benchmark --file output/benchmark.json
    python -m main.validate --type image_pair --file output/image_pairs.json --output-dir validation_reports
    python -m main.validate --type cot --file output/cot_data.json --config custom_config.yaml
    python -m main.validate --type sdf_caption --file output/sdf_caption.jsonl
"""

import asyncio
import argparse
import sys
from pathlib import Path

from src.config import OmniGenConfig
from src.core import ValidatorEngine
from src.validators import ValidatorResult
from .common import (
    add_common_validation_args,
    load_config_for_type,
    apply_validation_overrides,
    print_config_summary,
    handle_error,
    validate_data_type,
    validate_file_exists,
)


async def run_validation(config: OmniGenConfig, data_type: str, file_path: str) -> bool:
    """Run the validation process."""
    try:
        # Validate input file exists
        validate_file_exists(file_path, "data file")

        # Create validator engine
        engine = ValidatorEngine(config)

        # Run validation
        print(f"\n🔍 Starting {data_type} validation...")
        print(f"   Validating file: {file_path}")

        report_data = await engine.validate_file(file_path, data_type)

        if report_data and "validation_results" in report_data:
            # Extract statistics from report
            stats = report_data.get("statistics", {})
            validation_results = report_data["validation_results"]

            total_samples = stats.get("total_samples", len(validation_results))
            valid_samples = stats.get("valid_samples", 0)
            invalid_samples = total_samples - valid_samples
            avg_score = stats.get("average_score", 0)

            # Print summary
            print(f"\n📊 Validation Summary:")
            print(f"   Total samples: {total_samples}")
            print(f"   Valid samples: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
            print(f"   Invalid samples: {invalid_samples} ({invalid_samples/total_samples*100:.1f}%)")
            print(f"   Average score: {avg_score:.2f}/10")

            # Show quality distribution from statistics
            severity_counts = stats.get("severity_distribution", {})
            excellent = severity_counts.get("excellent", 0)
            good = severity_counts.get("good", 0)
            acceptable = severity_counts.get("acceptable", 0)
            poor = severity_counts.get("poor", 0)

            print(f"\n📈 Quality Distribution:")
            print(f"   Excellent (≥8.0): {excellent} ({excellent/total_samples*100:.1f}%)")
            print(f"   Good (7.0-7.9): {good} ({good/total_samples*100:.1f}%)")
            print(f"   Acceptable (5.0-6.9): {acceptable} ({acceptable/total_samples*100:.1f}%)")
            print(f"   Poor (<5.0): {poor} ({poor/total_samples*100:.1f}%)")

            # Show common issues from statistics
            issue_counts = stats.get("common_issues", {})
            if issue_counts:
                print(f"\n⚠️  Common Issues:")
                # Sort by count and take top 5
                sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for issue, count in sorted_issues:
                    print(f"   • {issue}: {count} occurrences")

            # Show detailed results if verbose
            if config.validation_config.verbose:
                print(f"\n📋 Detailed Results:")
                for i, result in enumerate(validation_results[:5], 1):  # Show first 5
                    print(f"   Sample {i}: {result.get('sample_id', f'sample_{i}')}")
                    print(
                        f"     Score: {result.get('overall_score', 0):.2f}/10 ({result.get('severity_level', 'unknown')})"
                    )
                    print(f"     Valid: {'✅' if result.get('is_valid', False) else '❌'}")
                    if result.get('issues'):
                        print(f"     Issues: {', '.join(result['issues'][:3])}")
                    if result.get('metric_scores'):
                        top_metrics = sorted(result['metric_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
                        metrics_str = ', '.join(f"{k}={v:.1f}" for k, v in top_metrics)
                        print(f"     Top metrics: {metrics_str}")
                    print()

                if len(validation_results) > 5:
                    print(f"   ... and {len(validation_results) - 5} more samples")

            # Show output file locations
            output_dir = config.validation_config.output_dir
            print(f"\n💾 Validation reports saved to:")
            print(f"   Directory: {output_dir}")
            print(f"   JSON report: {output_dir}/validation_report.json")
            print(f"   Summary: {output_dir}/validation_summary.txt")

            # Determine success based on validation quality
            success_rate = valid_samples / total_samples if total_samples > 0 else 0
            if success_rate >= 0.8:  # 80% or more valid samples
                print(f"\n✅ Validation completed successfully!")
                return True
            elif success_rate >= 0.5:  # 50-79% valid samples
                print(f"\n⚠️  Validation completed with warnings (low success rate: {success_rate:.1%})")
                return True
            else:  # Less than 50% valid samples
                print(f"\n❌ Validation failed (very low success rate: {success_rate:.1%})")
                return False
        else:
            print(f"\n❌ Validation failed - no results generated")
            return False

    except Exception as e:
        handle_error(e, "validation")
        return False


def print_data_type_info(data_type: str) -> None:
    """Print information about what will be validated for this data type."""
    validations = {
        "benchmark": [
            "Question quality and clarity",
            "Answer correctness",
            "Choice quality and distractors",
            "Image relevance to question",
            "Overall Q&A structure",
        ],
        "image_pair": [
            "Image file validity and format",
            "Text description quality",
            "Image-text semantic relevance",
            "Scientific accuracy",
            "Proper formatting",
        ],
        "qa_pair": [
            "Conversation structure and format",
            "Speaker alternation pattern",
            "Image relevance to conversations",
            "Question-answer flow and coherence",
        ],
        "cot": [
            "Question quality and clarity",
            "Solution completeness and accuracy",
            "Reasoning trajectory quality",
            "Model attempt quality",
            "Image relevance to reasoning (multimodal only)",
            "Logical flow and coherence",
        ],
        "sdf_caption": [
            "JSON caption structure",
            "Molecular formula grounding",
            "Numeric consistency with verified facts",
            "Prompt and dataset meta-language leakage",
        ],
    }

    print(f"\n🔍 Validation checks for {data_type} data:")
    for check in validations.get(data_type, ["General data structure validation"]):
        print(f"   • {check}")


def main():
    """Main entry point for unified validation."""
    parser = argparse.ArgumentParser(
        description="OmniGen Unified Data Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m main.validate --type benchmark --file output/benchmark.json
  python -m main.validate --type image_pair --file output/image_pairs.json --verbose
  python -m main.validate --type cot --file cot_data.json --output-dir custom_validation
  python -m main.validate --type sdf_caption --file output/sdf_caption.jsonl
        """,
    )

    # Add common validation arguments
    add_common_validation_args(parser)

    args = parser.parse_args()

    try:
        validate_data_type(args.type)

        # Load configuration (use default if not specified)
        config = load_config_for_type(args.type, args.config)

        # Apply overrides
        apply_validation_overrides(config, args)

        # Print information about what will be validated
        print_data_type_info(args.type)

        # Print configuration summary
        print_config_summary(config, args.type, "validation")

        # Run validation
        success = asyncio.run(run_validation(config, args.type, args.file))

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        handle_error(e, "argument parsing or configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
