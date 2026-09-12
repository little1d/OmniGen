import json
import os

import httpx
import pytest

from src.config import ModelConfig
from src.models import ModelHub, create_model


def test_text_request_and_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["ak"] == "test-ak"
        assert request.headers["X-TT-LOGID"] == "request-log-id"
        payload = json.loads(request.content)
        assert payload == {
            "stream": False,
            "model": "gpt-test",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is 1+1?"}],
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"X-TT-LOGID": "response-log-id"},
            json={
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {
                            "content": "2",
                            "reasoning_content": "One plus one equals two.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = ModelHub(
            model_name="gpt-test",
            ak="test-ak",
            endpoint="https://example.test/crawl",
            log_id="request-log-id",
            client=client,
        )
        result = model.generate_with_metadata("What is 1+1?", max_out_len=128)

    assert result["content"] == "2"
    assert result["reasoning_content"] == "One plus one equals two."
    assert result["usage"]["completion_tokens"] == 1
    assert result["log_id"] == "response-log-id"


def test_multimodal_request_supports_paths_and_urls(tmp_path):
    image_path = tmp_path / "molecule.jpg"
    image_path.write_bytes(b"jpeg-data")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        messages = payload["messages"]
        assert messages[0]["content"] == [{"type": "text", "text": "System prompt"}]
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "text", "text": "Describe the molecule"}
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert user_content[1]["image_url"]["detail"] == "auto"
        assert user_content[2]["image_url"] == {
            "url": "https://example.test/molecule.png",
            "detail": "high",
        }
        return httpx.Response(
            200,
            json={
                "data": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": [{"type": "text", "text": "A molecule."}]
                                }
                            }
                        ]
                    }
                )
            },
        )

    prompt = [
        {"role": "system", "content": "System prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the molecule"},
                {"type": "image_url", "image_url": {"url": str(image_path)}},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.test/molecule.png",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = ModelHub(
            model_name="gpt-test",
            ak="test-ak",
            endpoint="https://example.test/crawl",
            client=client,
        )
        assert model.generate(prompt) == "A molecule."


def test_errors_do_not_expose_ak():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid credential"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = ModelHub(
            model_name="gpt-test",
            ak="secret-value",
            endpoint="https://example.test/crawl",
            log_id="safe-log-id",
            client=client,
        )
        with pytest.raises(RuntimeError) as error:
            model.generate("test")

    assert "status=401" in str(error.value)
    assert "safe-log-id" in str(error.value)
    assert "secret-value" not in str(error.value)


def test_business_error_is_reported_with_log_id():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-TT-LOGID": "response-log-id"},
            json={"code": 1001, "message": "request rejected"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = ModelHub(
            model_name="gpt-test",
            ak="test-ak",
            endpoint="https://example.test/crawl",
            client=client,
        )
        with pytest.raises(RuntimeError, match="code=1001.*response-log-id"):
            model.generate("test")


def test_image_limit_and_endpoint_validation():
    images = [{"type": "image_url", "image_url": {"url": f"https://example.test/{i}.png"}} for i in range(11)]
    model = ModelHub(
        model_name="gpt-test",
        ak="test-ak",
        endpoint="https://example.test/crawl",
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
    )
    with pytest.raises(ValueError, match="at most 10 images"):
        model.generate({"text": "test", "images": images})
    model.client.close()

    with pytest.raises(ValueError, match="configuration invalid"):
        ModelHub(
            model_name="gpt-test",
            endpoint="https://example.test/crawl?ak=do-not-embed",
            ak="test-ak",
        )


def test_model_factory_and_config_defaults():
    model = create_model(
        "modelhub",
        {
            "model_name": "gpt-test",
            "ak": "test-ak",
            "endpoint": "https://example.test/crawl",
            "client": httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
        },
    )
    assert isinstance(model, ModelHub)
    model.client.close()

    config = ModelConfig()
    assert config.modelhub_endpoint == os.getenv("MODELHUB_ENDPOINT")
