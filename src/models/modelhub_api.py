import json
import os
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base_api import BaseAPIModel


class ModelHub(BaseAPIModel):
    """Chat client for the ModelHub HTTP API."""

    def __init__(
        self,
        model_name: str | None = None,
        ak: str | None = None,
        endpoint: str | None = None,
        log_id: str | None = None,
        timeout: float = 120.0,
        temperature: float | None = None,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name or os.getenv("MODELHUB_MODEL_NAME") or "",
            api_key=ak or os.getenv("MODELHUB_AK"),
            base_url=endpoint or os.getenv("MODELHUB_ENDPOINT"),
            **kwargs,
        )
        if not self.validate_config():
            raise ValueError(
                "ModelHub configuration invalid. "
                "Set MODELHUB_MODEL_NAME, MODELHUB_AK, and MODELHUB_ENDPOINT."
            )

        self.log_id = log_id or os.getenv("MODELHUB_LOG_ID")
        self.temperature = float(temperature) if temperature is not None else None
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=float(timeout),
            follow_redirects=True,
            transport=httpx.HTTPTransport(retries=2),
        )

    def validate_config(self) -> bool:
        if not super().validate_config() or not self.base_url:
            return False
        endpoint = urlsplit(self.base_url)
        return (
            endpoint.scheme in {"http", "https"}
            and bool(endpoint.netloc)
            and not endpoint.query
            and "${" not in self.base_url
        )

    def generate(self, prompt: Any, max_out_len: int = 512) -> str:
        return self.generate_with_metadata(prompt, max_out_len)["content"]

    def generate_with_reasoning(self, prompt: Any, max_out_len: int = 512) -> dict[str, str]:
        result = self.generate_with_metadata(prompt, max_out_len)
        return {
            "content": result["content"],
            "reasoning_content": result["reasoning_content"],
        }

    def generate_with_metadata(self, prompt: Any, max_out_len: int = 512) -> dict[str, Any]:
        if max_out_len <= 0:
            raise ValueError("max_out_len must be positive")

        log_id = self.log_id or uuid.uuid4().hex
        payload: dict[str, Any] = {
            "stream": False,
            "model": self.model_name,
            "max_tokens": max_out_len,
            "messages": self._prepare_messages(prompt),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        try:
            response = self.client.post(
                self.base_url,
                params={"ak": self.api_key},
                headers={"Content-Type": "application/json", "X-TT-LOGID": log_id},
                json=payload,
            )
        except httpx.HTTPError as error:
            raise RuntimeError(
                f"ModelHub request failed ({type(error).__name__}, log_id={log_id})"
            ) from None

        log_id = response.headers.get("X-TT-LOGID", log_id)
        if response.is_error:
            raise RuntimeError(f"ModelHub request failed (status={response.status_code}, log_id={log_id})")

        try:
            body = response.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"ModelHub returned non-JSON data (log_id={log_id})") from None
        if not isinstance(body, dict):
            raise RuntimeError(f"ModelHub returned an invalid response (log_id={log_id})")
        if body.get("code") not in (None, 0, "0", 200, "200"):
            raise RuntimeError(f"ModelHub returned an error (code={body['code']}, log_id={log_id})")

        data = body.get("data", body)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise RuntimeError(f"ModelHub returned invalid response data (log_id={log_id})") from None
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"ModelHub response has no completion (log_id={log_id})") from None

        return {
            "content": self._content_text(message.get("content")),
            "reasoning_content": self._content_text(
                message.get("reasoning_content") or message.get("reasoning")
            ),
            "model": data.get("model") or body.get("model") or self.model_name,
            "usage": data.get("usage") or body.get("usage") or {},
            "finish_reason": choice.get("finish_reason"),
            "log_id": log_id,
        }

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "ModelHub":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
