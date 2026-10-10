"""ADK-коллбэки, превращающие JSON-конверт в человекочитаемый ответ.

Корневой агент по контракту отвечает строгим JSON-конвертом
`ContentEnvelope` (см. [`ENVELOPE_FORMAT`](tutor_agent/prompts.py:37) и
[`ContentEnvelope`](tutor_agent/schemas.py:42)). Встроенный `adk web` печатает
текст ответа модели «как есть» и не может ничего знать про конверт — поэтому в
браузере был виден сырой JSON.

[`render_envelope_after_model()`](tutor_agent/callbacks.py:120) — это
`after_model_callback`, который перехватывает ответ модели и:

1. Разбирает конверт из ответа модели.
2. Подменяет содержимое ответа на «человеческий» markdown-текст (поле `text`
   конверта) — именно его показывают и `adk web`, и наш фронтенд.
3. Кладёт полный структурный конверт (в т.ч. `payload` квиза) в состояние
   события под ключом `temp:tutor_envelope`. Этот ключ попадает в
   `actions.stateDelta` SSE-события (см. `public_event(...).model_dump_json` в
   `google.adk.cli.api_server`), поэтому фронтенд получает структурные данные
   отдельным каналом и отрисовывает интерактивные элементы (варианты квиза).
   Префикс `temp:` не персистится в историю сессии.

Если ответ не является конвертом (обычный markdown) или содержит вызовы
инструментов — коллбэк ничего не меняет и возвращает `None`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

# Ключ состояния, в котором структурный конверт уходит фронтенду отдельным
# каналом (`actions.stateDelta` SSE-события). `temp:` — не персистится.
ENVELOPE_STATE_KEY = "temp:tutor_envelope"

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_FIRST_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _visible_text(llm_response: LlmResponse) -> str:
    """Собирает текст обычных (не «мысленных») частей ответа модели."""
    if not llm_response.content or not llm_response.content.parts:
        return ""
    return "".join(
        part.text
        for part in llm_response.content.parts
        if not getattr(part, "thought", False) and part.text
    )


def _has_tool_parts(llm_response: LlmResponse) -> bool:
    """Есть ли в ответе вызовы/ответы инструментов (их нельзя терять)."""
    if not llm_response.content or not llm_response.content.parts:
        return False
    return any(
        part.function_call is not None or part.function_response is not None
        for part in llm_response.content.parts
    )


def _parse_envelope(raw: str) -> dict[str, Any] | None:
    """Устойчиво извлекает конверт `{type, text, payload, difficulty}`.

    Снимает markdown-обёртку ```` ```json ```` и лишний текст вокруг объекта.
    Возвращает `None`, если ответ не является конвертом (например, обычный
    markdown или незавершённый стриминговый JSON-фрагмент).
    """
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    if "{" not in text:
        return None
    match = _FIRST_JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return data
    return None


def _looks_like_json(raw: str) -> bool:
    """Похож ли фрагмент на (возможно, незавершённый) JSON-объект."""
    stripped = (raw or "").strip()
    return stripped.startswith("{") or stripped.startswith("```")


def render_envelope_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """`after_model_callback`: конверт → markdown + структурный канал.

    Returns:
        Новый `LlmResponse` с «человеческим» текстом, либо `None`, если
        ответ менять не нужно (не конверт / есть вызовы инструментов).
    """
    raw = _visible_text(llm_response)
    if not raw:
        return None

    # Если модель в этом ходе вызывает инструмент — контент не трогаем.
    if _has_tool_parts(llm_response):
        return None

    envelope = _parse_envelope(raw)
    if envelope is not None:
        # Структурные данные — фронтенду отдельным каналом.
        try:
            callback_context.state[ENVELOPE_STATE_KEY] = envelope
        except Exception:  # pragma: no cover - защита от несериализуемых данных
            logger.warning(
                "Не удалось сохранить конверт в state (ключ %s)",
                ENVELOPE_STATE_KEY,
                exc_info=True,
            )

        human_text = envelope.get("text", "")
        new_content = types.Content(role="model", parts=[types.Part(text=human_text)])
        return llm_response.model_copy(update={"content": new_content})

    # Незавершённый/невалидный JSON (частичный стриминговый чанк): прячем его,
    # чтобы в adk web не мигал сырой JSON. Финальный чанк распарсится выше.
    if llm_response.partial and _looks_like_json(raw):
        empty = types.Content(role="model", parts=[])
        return llm_response.model_copy(update={"content": empty})

    return None
