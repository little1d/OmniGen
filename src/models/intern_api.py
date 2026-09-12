import os
from typing import Any

from openai import OpenAI

from .base_api import BaseAPIModel


class InternAPI(BaseAPIModel):
    """Shared OpenAI-compatible client for Intern models."""

    env_prefix = "INTERN"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        client: Any = None,
        **kwargs: Any,
    ):
        prefix = self.env_prefix
        super().__init__(
            model_name=model_name or os.getenv(f"{prefix}_MODEL_NAME") or "",
            api_key=api_key or os.getenv(f"{prefix}_API_KEY"),
            base_url=base_url or os.getenv(f"{prefix}_BASE_URL"),
            **kwargs,
        )
        if not self.validate_config():
            raise ValueError(
                f"{self.__class__.__name__} configuration invalid. "
                f"Set {prefix}_MODEL_NAME, {prefix}_API_KEY, and {prefix}_BASE_URL."
            )
        self.temperature = float(temperature)
        self.client = client or OpenAI(api_key=self.api_key, base_url=self.base_url)

    def validate_config(self) -> bool:
        return super().validate_config() and bool(
            self.base_url and self.base_url.strip() and "${" not in self.base_url
        )

    def generate(
        self,
        prompt: Any,
        max_out_len: int = 512,
        return_reasoning: bool = False,
    ) -> str | dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._prepare_messages(prompt),
                max_tokens=max_out_len,
                temperature=self.temperature,
            )
        except Exception as error:
            raise RuntimeError(f"{self.__class__.__name__} API call failed: {error}") from error

        message = response.choices[0].message
        content = message.content or ""
        if not return_reasoning:
            return content
        return {
            "content": content,
            "reasoning_content": getattr(message, "reasoning_content", "") or "",
        }

    def generate_with_reasoning(self, prompt: Any, max_out_len: int = 512) -> dict[str, str]:
        return self.generate(prompt, max_out_len, return_reasoning=True)


class InternS1(InternAPI):
    env_prefix = "INTERNS1"


class InternVL(InternAPI):
    env_prefix = "INTERNVL"
