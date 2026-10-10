"""ADK-коллбэки, превращающие JSON-конверт в человекочитаемый ответ.

Корневой агент по контракту отвечает строгим JSON-конвертом
`ContentEnvelope` (см. [`ENVELOPE_FORMAT`](tutor_agent/prompts.py:37) и
[`ContentEnvelope`](tutor_agent/schemas.py:42)). Встроенный `adk web` печатает
текст ответа модели «как есть» и не может ничего знать про конверт — поэтому в
браузере был виден сырой JSON.

[`render_envelope_after_model()`](tutor_agent/callbacks.py:196) — это
`after_model_callback`, который перехватывает ответ модели и:

1. Разбирает конверт(ы) из ответа модели. Модель иногда (особенно после вызова
   инструмента, например `generate_summary`) шлёт **несколько** конвертов в
   одном ответе — их балансированно извлекаем и сводим в один
   ([`_merge_envelopes()`](tutor_agent/callbacks.py:120)), а не пытаемся
   распарсить склеенный JSON жадно.
2. Подменяет содержимое ответа на «человеческий» markdown-текст (поле `text`
   конверта) — именно его показывают и `adk web`, и наш фронтенд.
3. Кладёт полный структурный конверт (в т.ч. `payload` квиза) в состояние
   события под ключом `temp:tutor_envelope`. Этот ключ попадает в
   `actions.stateDelta` SSE-события (см. `public_event(...).model_dump_json` в
   `google.adk.cli.api_server`), поэтому фронтенд получает структурные данные
   отдельным каналом и отрисовывает интерактивные элементы (варианты квиза).
   Префикс `temp:` не персистится в историю сессии.
4. В стриминговом режиме прячет промежуточные JSON-чанки: флаг
   `temp:tutor_envelope_streaming` взводится на первом чанке с `{` и держится
   до финального ответа, чтобы в UI не мигал сырой JSON.

Если ответ не является конвертом (обычный markdown) или содержит вызовы
инструментов — коллбэк ничего не меняет и возвращает `None`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

# Ключ состояния, в котором структурный конверт уходит фронтенду отдельным
# каналом (`actions.stateDelta` SSE-события). `temp:` — не персистится.
ENVELOPE_STATE_KEY = "temp:tutor_envelope"

# Флаг «сейчас стримится JSON-конверт». Пока он взведён, промежуточные
# стриминговые чанки прячем, чтобы в UI не мигал сырой JSON. `temp:` —
# не персистится в историю сессии.
ENVELOPE_STREAM_KEY = "temp:tutor_envelope_streaming"

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


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


def _iter_json_objects(raw: str) -> Iterator[dict[str, Any]]:
    """Перебирает JSON-объекты верхнего уровня в тексте модели.

    Устойчив к «мусору» вокруг (markdown-обёртка, пояснения) и, что важно,
    к нескольким объектам подряд. Жадный `\\{.*\\}` на двух конвертах даёт
    невалидный JSON и ломает разбор — поэтому считаем баланс скобок с учётом
    строк, начиная от каждой `{`. Незакрытый объект (частичный стриминговый
    чанк) просто завершает перебор.
    """
    text = _FENCE_RE.sub("", raw or "")
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        end: int | None = None
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return  # незакрытый объект — вероятно, частичный стриминговый чанк
        try:
            obj = json.loads(text[i : end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            yield obj
        i = end + 1


def _parse_envelopes(raw: str) -> list[dict[str, Any]]:
    """Извлекает все конверты `{type, text, payload, difficulty}` из ответа."""
    return [
        obj for obj in _iter_json_objects(raw) if isinstance(obj.get("text"), str)
    ]


def _merge_envelopes(envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводит несколько конвертов в один (модель иногда шлёт их подряд).

    Особенно часто это бывает после вызова инструмента (например, сначала
    конверт `theory`, затем `quiz`). Склеиваем тексты, а `type`/`payload`
    берём у квиза (если он есть с непустым payload), иначе у последнего
    конверта. Текст самого квиза не дублируем — его рисует карточка квиза.
    """
    if len(envelopes) == 1:
        return envelopes[0]

    quiz = next(
        (
            env
            for env in envelopes
            if env.get("type") == "quiz"
            and isinstance(env.get("payload"), dict)
            and env["payload"]
        ),
        None,
    )

    texts = [
        env["text"].strip()
        for env in envelopes
        if env.get("text", "").strip() and env is not quiz
    ]
    if not texts:
        texts = [env["text"].strip() for env in envelopes if env.get("text", "").strip()]

    merged: dict[str, Any] = {
        "type": envelopes[-1].get("type", "theory"),
        "text": "\n\n".join(texts),
        "payload": {},
        "difficulty": envelopes[-1].get("difficulty", "medium"),
    }
    if quiz is not None:
        merged["type"] = "quiz"
        merged["payload"] = quiz["payload"]
        merged["difficulty"] = quiz.get("difficulty", merged["difficulty"])
    else:
        last_payload = envelopes[-1].get("payload")
        if isinstance(last_payload, dict):
            merged["payload"] = last_payload
    return merged


def _stream_flag(callback_context: CallbackContext) -> bool:
    """Взведён ли флаг «стримится JSON-конверт»."""
    try:
        return bool(callback_context.state.get(ENVELOPE_STREAM_KEY))
    except Exception:  # pragma: no cover - защита от нестандартного state
        return False


def _set_stream_flag(callback_context: CallbackContext, value: bool) -> None:
    """Взводит/снимает флаг стриминга конверта (не критично при ошибке state)."""
    try:
        callback_context.state[ENVELOPE_STREAM_KEY] = value
    except Exception:  # pragma: no cover - защита от нестандартного state
        pass


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

    # Стриминговые чанки: пока модель отдаёт JSON-конверт, прячем промежуточные
    # фрагменты (флаг взводится на первом чанке и держится до финального),
    # чтобы в UI не мигал сырой JSON.
    if llm_response.partial:
        streaming = _stream_flag(callback_context)
        if _looks_like_json(raw):
            _set_stream_flag(callback_context, True)
            streaming = True
        if streaming:
            empty = types.Content(role="model", parts=[])
            return llm_response.model_copy(update={"content": empty})
        return None

    # Финальный ответ — снимаем флаг стриминга.
    _set_stream_flag(callback_context, False)

    envelopes = _parse_envelopes(raw)
    if envelopes:
        envelope = _merge_envelopes(envelopes)
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

    # Не распарсили конверт (обычный markdown) — оставляем как есть.
    return None
