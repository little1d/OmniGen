from types import SimpleNamespace

import pytest

from src.models.intern_api import InternS1, InternVL


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="answer", reasoning_content="reasoning")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


@pytest.mark.parametrize("model_class", [InternS1, InternVL])
def test_intern_models_share_the_same_client(model_class):
    client = FakeClient()
    model = model_class(
        model_name="intern-test",
        api_key="test-key",
        base_url="https://example.test/v1",
        client=client,
    )

    result = model.generate_with_reasoning("hello", max_out_len=32)

    assert result == {"content": "answer", "reasoning_content": "reasoning"}
    assert client.chat.completions.request["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert client.chat.completions.request["max_tokens"] == 32
