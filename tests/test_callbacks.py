"""Тесты after_model_callback: конверт → markdown + структурный канал."""

from __future__ import annotations

import json
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from tutor_agent.callbacks import (
    ENVELOPE_STATE_KEY,
    render_envelope_after_model,
)


class _FakeContext:
    """Минимальная заглушка CallbackContext: нужен только `.state`."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}


def _response(text: str, *, partial: bool | None = None) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        partial=partial,
    )


def _envelope_text(**overrides: Any) -> str:
    data = {
        "type": "theory",
        "text": "# Заголовок\n\nПояснение.",
        "payload": {},
        "difficulty": "medium",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_envelope_replaced_with_human_text() -> None:
    ctx = _FakeContext()
    result = render_envelope_after_model(ctx, _response(_envelope_text()))

    assert result is not None
    assert result.content is not None
    assert result.content.parts
    assert result.content.parts[0].text == "# Заголовок\n\nПояснение."


def test_envelope_saved_to_state_channel() -> None:
    ctx = _FakeContext()
    render_envelope_after_model(ctx, _response(_envelope_text()))

    assert ENVELOPE_STATE_KEY in ctx.state
    assert ctx.state[ENVELOPE_STATE_KEY]["type"] == "theory"


def test_quiz_payload_saved_for_frontend() -> None:
    ctx = _FakeContext()
    raw = _envelope_text(
        type="quiz",
        text="Выбери верный вариант.",
        payload={"question": "2+2?", "options": ["3", "4", "5", "6"]},
    )
    result = render_envelope_after_model(ctx, _response(raw))

    assert result is not None
    stored = ctx.state[ENVELOPE_STATE_KEY]
    assert stored["type"] == "quiz"
    assert stored["payload"]["options"] == ["3", "4", "5", "6"]


def test_envelope_wrapped_in_markdown_fence() -> None:
    ctx = _FakeContext()
    raw = "```json\n" + _envelope_text() + "\n```"
    result = render_envelope_after_model(ctx, _response(raw))

    assert result is not None
    assert result.content.parts[0].text == "# Заголовок\n\nПояснение."


def test_plain_markdown_is_untouched() -> None:
    ctx = _FakeContext()
    result = render_envelope_after_model(ctx, _response("# Обычный ответ\n\nТекст."))

    assert result is None
    assert ENVELOPE_STATE_KEY not in ctx.state


def test_tool_call_response_is_untouched() -> None:
    ctx = _FakeContext()
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="web_search", args={})
                )
            ],
        )
    )
    assert render_envelope_after_model(ctx, resp) is None
    assert ENVELOPE_STATE_KEY not in ctx.state


def test_partial_json_chunk_is_hidden() -> None:
    ctx = _FakeContext()
    result = render_envelope_after_model(
        ctx, _response('{"type": "theory", "text": "Прив', partial=True)
    )

    assert result is not None
    assert result.content is not None
    assert result.content.parts == []


def test_partial_plain_text_is_untouched() -> None:
    ctx = _FakeContext()
    # Стриминговый чанк обычного markdown не должен прятаться.
    assert render_envelope_after_model(ctx, _response("Привет", partial=True)) is None


def test_two_envelopes_in_one_response_are_merged() -> None:
    ctx = _FakeContext()
    # Регрессия: после вызова инструмента (generate_summary) модель шлёт два
    # конверта подряд — прежний жадный `\\{.*\\}` давал невалидный JSON, и
    # сырой JSON протекал в чат.
    theory = _envelope_text(type="theory", text="Разбор темы.")
    quiz = _envelope_text(
        type="quiz",
        text="Проверим понимание.",
        payload={"question": "2+2?", "options": ["3", "4", "5", "6"]},
    )
    result = render_envelope_after_model(ctx, _response(theory + "\n" + quiz))

    assert result is not None
    # Текст — человекочитаемый (конкатенация), без фигурных скобок.
    text = result.content.parts[0].text
    assert "{" not in text
    assert "Разбор темы." in text

    stored = ctx.state[ENVELOPE_STATE_KEY]
    assert stored["type"] == "quiz"
    assert stored["payload"]["options"] == ["3", "4", "5", "6"]


def test_streaming_json_hidden_until_final_by_flag() -> None:
    ctx = _FakeContext()
    # Первый чанк конверта — взводит флаг и прячется.
    first = render_envelope_after_model(
        ctx, _response('{"type": "theory", "te', partial=True)
    )
    assert first is not None and first.content.parts == []
    # Следующий чанк уже не начинается с `{`, но флаг держится — тоже прячем.
    second = render_envelope_after_model(ctx, _response('xt": "Прив', partial=True))
    assert second is not None and second.content.parts == []


def test_envelope_text_with_braces_parsed() -> None:
    ctx = _FakeContext()
    raw = json.dumps(
        {
            "type": "theory",
            "text": "Функция задаётся как y = {0} при x > 0.",
            "payload": {},
            "difficulty": "medium",
        },
        ensure_ascii=False,
    )
    result = render_envelope_after_model(ctx, _response(raw))

    assert result is not None
    assert result.content.parts[0].text == "Функция задаётся как y = {0} при x > 0."


def test_empty_response_returns_none() -> None:
    ctx = _FakeContext()
    assert render_envelope_after_model(ctx, LlmResponse()) is None
