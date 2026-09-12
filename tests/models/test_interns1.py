import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env file is loaded
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

from src.models.intern_api import InternS1


class TestInternS1:
    def test_config_loading(self):
        from src.config import ModelConfig

        config = ModelConfig()

        # Check if config values exist (from environment variables or defaults)
        print(f"API Key status: {'Set' if config.interns1_api_key else 'Not Set'}")
        print(f"Base URL: {config.interns1_base_url}")
        print(f"Model Name: {config.interns1_model_name}")

        assert config.interns1_api_key == os.getenv("INTERNS1_API_KEY")
        assert config.interns1_base_url == os.getenv("INTERNS1_BASE_URL")
        assert config.interns1_model_name == os.getenv("INTERNS1_MODEL_NAME")

    @pytest.mark.skipif(
        not all(os.getenv(name) for name in ("INTERNS1_API_KEY", "INTERNS1_BASE_URL", "INTERNS1_MODEL_NAME")),
        reason="Requires complete InternS1 API configuration",
    )
    def test_model_real_api_call(self):
        """Test real API call"""
        # Validate environment variable
        api_key = os.getenv("INTERNS1_API_KEY")
        assert api_key is not None, "INTERNS1_API_KEY environment variable not set"

        # Create model instance
        model = InternS1()

        # Print actual model information used
        print(f"Model used: {model.model_name}")
        print(f"API Endpoint: {model.base_url}")

        # Test simple text generation
        try:
            response = model.generate("Hello, please reply briefly", max_out_len=200)  # 增加token数量
            print(f"Raw response: '{response}'")
            print(f"Response length: {len(response)}")
        except Exception as e:
            print(f"API call failed with error: {e}")
            raise

        # Test with reasoning content
        try:
            response_with_reasoning = model.generate_with_reasoning(
                "Hello, please reply briefly", max_out_len=200
            )  # 增加token数量
            print(f"Content: '{response_with_reasoning['content']}'")
            print(f"Reasoning: '{response_with_reasoning['reasoning_content']}'")
        except Exception as e:
            print(f"Reasoning API call failed with error: {e}")
            raise

        # Validate response - check if we have either content or reasoning_content
        has_content = len(response) > 0
        has_reasoning = len(response_with_reasoning.get('reasoning_content', '')) > 0

        assert (
            has_content or has_reasoning
        ), f"No content received. Content: '{response}', Reasoning: '{response_with_reasoning.get('reasoning_content', '')}'"

        # Final summary
        print(f"\n=== Final Results ===")
        print(f"Content: '{response}' (length: {len(response)})")
        print(
            f"Reasoning: '{response_with_reasoning.get('reasoning_content', '')}' (length: {len(response_with_reasoning.get('reasoning_content', ''))})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
