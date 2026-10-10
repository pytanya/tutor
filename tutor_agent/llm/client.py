"""Прямые вызовы RouterAI (OpenAI-совместимый API) для утилитарных задач.

ADK-агенты используют `LiteLlm` для диалога, но для отдельных «одноразовых»
вызовов (выжимка, генерация квиза, веб-поиск) удобнее обращаться к API
напрямую через `httpx`. Это позволяет точнее управлять параметрами
(`plugins`, `max_tokens`) и не гонять лишние события через сессию агента.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import get_settings

_REQUEST_TIMEOUT = 60.0


class RouterAIError(RuntimeError):
    """Ошибка при обращении к RouterAI."""


async def chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    plugins: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Выполняет запрос к `/chat/completions` RouterAI.

    Args:
        model: id модели RouterAI (например `deepseek/deepseek-v4-flash-0731`).
        messages: Список сообщений в формате OpenAI.
        max_tokens: Ограничение длины ответа.
        temperature: Температура сэмплирования.
        plugins: Плагины RouterAI (например веб-поиск `{"id": "web", ...}`).

    Returns:
        Полный JSON-ответ API.

    Raises:
        RouterAIError: При ошибке сети или не-2xx ответе.
    """
    settings = get_settings()
    if not settings.routerai_api_key:
        raise RouterAIError(
            "ROUTERAI_API_KEY не задан. Скопируйте .env.example в .env и "
            "укажите ключ RouterAI."
        )

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if plugins:
        payload["plugins"] = plugins

    url = f"{settings.routerai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.routerai_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:  # сеть/таймаут
        raise RouterAIError(f"Сетевая ошибка при обращении к RouterAI: {exc}") from exc

    if response.status_code >= 400:
        raise RouterAIError(
            f"RouterAI вернул {response.status_code}: {response.text[:500]}"
        )

    return response.json()


def extract_message_content(data: dict[str, Any]) -> str:
    """Извлекает текст ответа ассистента из JSON Chat Completion.

    Если провайдер вернул пустой `content`, но заполнил канал рассуждений
    (`reasoning`/`reasoning_content`) — это признак «рассуждающей» модели,
    которая не выдала финальный ответ. Такие модели (например
    `deepseek-v4-flash`) непригодны для инструментов, где ожидается готовый
    текст или JSON, поэтому сообщаем об этом явной ошибкой.
    """
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RouterAIError(f"Неожиданный формат ответа RouterAI: {data}") from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        raise RouterAIError(
            "Модель вернула только рассуждения (reasoning) без ответа в "
            "`content`. Для planner/fast используйте модель без reasoning-"
            "канала (например `openai/gpt-4o-mini` или `deepseek-v3.2`)."
        )
    return ""


def extract_citations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Извлекает веб-цитаты (`url_citation`) из ответа с плагином поиска."""
    try:
        annotations = data["choices"][0]["message"].get("annotations") or []
    except (KeyError, IndexError, TypeError):
        return []
    return [
        ann["url_citation"]
        for ann in annotations
        if isinstance(ann, dict) and ann.get("type") == "url_citation"
    ]
