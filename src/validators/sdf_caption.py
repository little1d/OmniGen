from typing import Any

from ..sdf_caption import validate_response
from .base import BaseValidator
from .registry import register_validator
from .result import ValidatorResult


@register_validator("sdf_caption")
class SDFCaptionValidator(BaseValidator):
    """Deterministically validate caption claims against verified facts."""

    def get_supported_data_type(self) -> str:
        return "sdf_caption"

    def get_required_fields(self) -> list[str]:
        return [
            "caption",
            "smiles",
            "facts",
            "sdf",
        ]

    async def validate_sample(self, sample: dict[str, Any]) -> ValidatorResult:
        issues = self._check_required_fields(sample)
        validation = validate_response(
            {"caption": sample.get("caption")},
            sample.get("facts", {}),
        )
        issues.extend(validation["errors"])

        metric_scores = {
            "format": 10.0
            if not any(
                "required field" in issue
                or "caption field" in issue
                or "non-empty" in issue
                for issue in issues
            )
            else 0.0,
            "formula_grounding": 10.0
            if not any("molecular formula" in issue for issue in issues)
            else 0.0,
            "numeric_grounding": 10.0
            if not any("decimal value" in issue for issue in issues)
            else 0.0,
            "content_hygiene": 10.0
            if not any("meta-language" in issue for issue in issues)
            else 0.0,
        }
        overall_score = sum(metric_scores.values()) / len(metric_scores)
        return self._create_result(
            sample=sample,
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=list(dict.fromkeys(issues)),
            metadata={
                "deterministic": True,
            },
        )
