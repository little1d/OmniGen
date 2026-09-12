import json
from pathlib import Path
from typing import List, Dict, Any
from ..config import OmniGenConfig
from ..models import create_model
from ..validators.registry import create_validator
from ..validators.result import ValidatorResult


class ValidatorEngine:
    """Validation engine - manages validation workflow"""

    def __init__(self, config: OmniGenConfig):
        self.config = config
        self.validation_config = config.validation_config
        self.model_client = None

    def _init_model_client(self):
        """Initialize model client - reuse generation engine logic"""
        model_type = self.config.model_type.lower()
        try:
            self.model_client = create_model(model_type, self.config.model_config)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize model client '{model_type}': {error}") from error

    async def validate_file(self, file_path: str, data_type: str) -> Dict[str, Any]:
        """Validate entire output file"""
        if not self.validation_config.enabled:
            print("Validation is disabled")
            return {"enabled": False}

        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            if path.suffix.lower() == '.jsonl':
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Validation input must be a JSON array or JSONL file")

        if self.validation_config.verbose:
            print(f"Validating {len(data)} samples from {file_path}")

        # Create validator for the data type
        validator = create_validator(data_type, self.model_client, self.validation_config)
        if validator is None:
            raise ValueError(f"No validator found for data type: {data_type}")
        if validator.requires_model:
            self._init_model_client()
            validator.model_client = self.model_client

        # Validate all samples
        results = []
        for i, sample in enumerate(data):
            sample["sample_id"] = sample.get("sample_id") or sample.get("id") or f"sample_{i}"

            if self.validation_config.verbose:
                print(f"Validating sample {i+1}/{len(data)}: {sample['sample_id']}")

            result = await validator.validate_sample(sample)
            results.append(result)

            if self.validation_config.verbose:
                print(f"  Score: {result.overall_score:.2f} ({result.severity_level})")

        # Calculate statistics
        stats = self._calculate_statistics(results)

        if self.validation_config.verbose:
            print(f"\nValidation Summary:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Valid samples: {stats['valid_samples']} ({stats['validity_rate']:.1%})")
            print(f"  Average score: {stats['average_score']:.2f}")

        # Generate report
        report_data = {
            "source_file": file_path,
            "data_type": data_type,
            "validation_results": [self._result_to_dict(r) for r in results],
            "statistics": stats,
            "config": {
                "overall_threshold": self.validation_config.overall_threshold,
                "critical_threshold": self.validation_config.critical_threshold,
            },
        }

        # Save reports
        self._save_reports(report_data, file_path, data_type)

        return report_data

    def _calculate_statistics(self, results: List[ValidatorResult]) -> Dict[str, Any]:
        """Calculate validation statistics"""
        total = len(results)
        if total == 0:
            return {}

        valid_count = sum(1 for r in results if r.is_valid)
        score_sum = sum(r.overall_score for r in results)

        # Group by severity level
        severity_counts = {}
        for result in results:
            level = result.severity_level
            severity_counts[level] = severity_counts.get(level, 0) + 1

        # Collect common issues
        issue_counts = {}
        for result in results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        return {
            "total_samples": total,
            "valid_samples": valid_count,
            "invalid_samples": total - valid_count,
            "validity_rate": valid_count / total,
            "average_score": score_sum / total,
            "severity_distribution": severity_counts,
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        }

    def _result_to_dict(self, result: ValidatorResult) -> Dict[str, Any]:
        """Convert ValidatorResult to dictionary"""
        return {
            "sample_id": result.sample_id,
            "data_type": result.data_type,
            "overall_score": result.overall_score,
            "metric_scores": result.metric_scores,
            "issues": result.issues,
            "warnings": result.warnings,
            "suggestions": result.suggestions,
            "is_valid": result.is_valid,
            "severity_level": result.severity_level,
            "metadata": result.metadata,
        }

    def _save_reports(self, report_data: Dict[str, Any], source_file: str, data_type: str):
        """Save validation reports"""
        output_dir = Path(self.validation_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate file names
        source_name = Path(source_file).stem

        # Save JSON report
        json_file = output_dir / f"{source_name}_{data_type}_validation.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # Generate summary report
        stats = report_data["statistics"]
        summary_file = output_dir / f"{source_name}_{data_type}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Validation Report Summary - {data_type.upper()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source file: {source_file}\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Valid samples: {stats['valid_samples']} ({stats['validity_rate']:.1%})\n")
            f.write(f"Average score: {stats['average_score']:.2f}\n\n")

            f.write("Severity distribution:\n")
            for level, count in stats['severity_distribution'].items():
                f.write(f"  {level}: {count}\n")

            if stats['common_issues']:
                f.write("\nCommon issues:\n")
                for issue, count in list(stats['common_issues'].items())[:5]:
                    f.write(f"  - {issue}: {count} times\n")

        if self.validation_config.verbose:
            print(f"Validation reports saved to: {output_dir}")
