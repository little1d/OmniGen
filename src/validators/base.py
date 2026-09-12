from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from .result import ValidatorResult
from ..config.config import ValidatorConfig


class BaseValidator(ABC):
    """Base class for all validators"""

    requires_model: bool = False

    def __init__(self, model_client, validation_config: ValidatorConfig):
        self.model_client = model_client
        self.validation_config = validation_config

    @abstractmethod
    async def validate_sample(self, sample: Dict[str, Any]) -> ValidatorResult:
        """Validate a single sample and return validation result"""
        pass

    @abstractmethod
    def get_supported_data_type(self) -> str:
        """Get the data type this validator supports"""
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields for this validator"""
        pass

    def _check_required_fields(self, sample: Dict[str, Any]) -> List[str]:
        """Check if sample has all required fields"""
        issues = []
        required_fields = self.get_required_fields()

        for field in required_fields:
            if field not in sample or sample[field] is None:
                issues.append(f"Missing required field: {field}")
            elif isinstance(sample[field], str) and not sample[field].strip():
                issues.append(f"Empty required field: {field}")

        return issues

    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average score"""
        total_weight = 0
        weighted_sum = 0

        for metric, score in scores.items():
            if metric in weights:
                weight = weights[metric]
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _create_result(
        self,
        sample: Dict[str, Any],
        overall_score: float,
        metric_scores: Dict[str, float],
        issues: List[str],
        warnings: List[str] = None,
        suggestions: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> ValidatorResult:
        """Helper method to create ValidatorResult"""
        is_valid = overall_score >= self.validation_config.overall_threshold and len(issues) == 0

        return ValidatorResult(
            sample_id=sample.get("sample_id", "unknown"),
            data_type=self.get_supported_data_type(),
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=issues,
            warnings=warnings or [],
            suggestions=suggestions or [],
            is_valid=is_valid,
            metadata=metadata or {},
        )
