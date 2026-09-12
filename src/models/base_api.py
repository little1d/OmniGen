import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

from .base import BaseModel


class BaseAPIModel(BaseModel):
    """Shared configuration and message formatting for chat APIs."""

    is_api: bool = True
    max_images: int = 10
    max_image_bytes: int = 5 * 1024 * 1024

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(path=model_name)
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    def validate_config(self) -> bool:
        values = (self.api_key, self.model_name)
        return all(value and value.strip() and "${" not in value for value in values)

    def _prepare_messages(self, prompt: Any) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]
        elif isinstance(prompt, dict):
            content: list[dict[str, Any]] = []
            if prompt.get("text"):
                content.append({"type": "text", "text": str(prompt["text"])})
            content.extend(self._image_part(image) for image in prompt.get("images", []))
            messages = [{"role": "user", "content": content}]
        else:
            raise TypeError("prompt must be text, a message list, or a mapping")

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        normalized = []
        image_count = 0
        for message in messages:
            if not isinstance(message, dict) or not message.get("role"):
                raise ValueError("each message must contain a role")
            content = message.get("content", "")
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if not isinstance(parts, list) or not parts:
                raise ValueError("message content must be non-empty text or a content list")

            normalized_parts = []
            for part in parts:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict) and part.get("type") == "text":
                    normalized_parts.append({"type": "text", "text": str(part.get("text", ""))})
                else:
                    normalized_parts.append(self._image_part(part))
                    image_count += 1
            normalized.append({"role": str(message["role"]), "content": normalized_parts})

        if image_count > self.max_images:
            raise ValueError(f"at most {self.max_images} images are allowed per request")
        return normalized

    def _image_part(self, image: Any) -> dict[str, Any]:
        detail = "auto"
        if isinstance(image, dict) and image.get("type") == "pil_image":
            image = image.get("image")
        elif isinstance(image, dict):
            value = image.get("image_url", image)
            if isinstance(value, dict):
                detail = value.get("detail", "auto")
                image = value.get("url")
            else:
                image = value

        if detail not in {"low", "high", "auto"}:
            raise ValueError("image detail must be low, high, or auto")
        if isinstance(image, Path):
            image = str(image)
        if isinstance(image, str):
            url = image if image.startswith(("http://", "https://", "data:")) else self._file_data_uri(image)
        else:
            url = self._pil_data_uri(image)
        return {"type": "image_url", "image_url": {"url": url, "detail": detail}}

    def _file_data_uri(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise ValueError(f"image file does not exist: {path}")
        if path.stat().st_size > self.max_image_bytes:
            raise ValueError("image exceeds the 5 MB limit")
        mime_type = mimetypes.guess_type(path.name)[0]
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"unsupported image format: {path.suffix or path.name}")
        value = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{value}"

    def _pil_data_uri(self, image: Any) -> str:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise ValueError(f"unsupported image input: {type(image).__name__}")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        value = buffer.getvalue()
        if len(value) > self.max_image_bytes:
            raise ValueError("image exceeds the 5 MB limit")
        return f"data:image/png;base64,{base64.b64encode(value).decode('ascii')}"

    @staticmethod
    def _content_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content)
