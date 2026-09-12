import os
import base64
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from .base_api import BaseAPIModel
from openai import OpenAI


class Gemini(BaseAPIModel):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Set default values for Gemini
        model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "google/gemini-2.5-pro")
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        base_url = base_url or os.getenv("GEMINI_BASE_URL")

        # Initialize parent class
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)

        # Validate configuration
        if not self.validate_config():
            raise ValueError("Gemini configuration invalid. Please set GEMINI_API_KEY environment variable.")

        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: Union[str, Dict[str, Any]], max_out_len: int = 512) -> str:
        """
        Generate response supporting text and multimodal input

        Args:
            prompt: Input prompt, can be:
                   - str: Pure text prompt
                   - Dict: Multimodal prompt, format:
                     {
                         "text": "Question text",
                         "images": [{"type": "image_url", "image_url": {"url": "data:..."}}]
                     }
            max_out_len: Maximum output length

        Returns:
            Generated response string
        """
        try:
            messages = self._prepare_messages(prompt)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_out_len,
                extra_body={},
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    def _prepare_messages(self, prompt: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare message format"""
        messages = []

        if isinstance(prompt, dict):
            # Multimodal input
            content = []

            # Add text content
            text = prompt.get("text", "")
            if text:
                content.append({"type": "text", "text": text})

            # Add image content
            images = prompt.get("images", [])
            for image_data in images:
                if isinstance(image_data, dict):
                    content.append(image_data)
                elif isinstance(image_data, str):
                    # If it's an image path, convert to base64 format
                    image_content = self._process_image_path(image_data)
                    if image_content:
                        content.append(image_content)

            messages.append({"role": "user", "content": content})

        else:
            # Pure text input
            text_content = str(prompt)
            messages.append({"role": "user", "content": [{"type": "text", "text": text_content}]})

        return messages

    def _process_image_path(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Process image path, convert to API required format"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Warning: image file does not exist: {image_path}")
                return None

            # Read image and convert to base64
            with open(path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Get image format
            image_format = path.suffix.lower().lstrip('.')
            if image_format == 'jpg':
                image_format = 'jpeg'

            return {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}

        except Exception as e:
            print(f"Failed to process image: {e}")
            return None

    def validate_config(self) -> bool:
        """Validate configuration"""
        return super().validate_config() and self.base_url is not None
