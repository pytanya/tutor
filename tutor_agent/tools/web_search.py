"""Инструмент веб-поиска через плагин RouterAI.

Встроенный `google_search` в ADK работает только на Gemini, поэтому поиск
реализован собственным инструментом поверх плагина веб-поиска RouterAI
(`{"id": "web"}`). Возвращает текст с ответом и список источников-цитат.
"""

from __future__ import annotations

from typing import Any

from ..llm.client import RouterAIError, chat_completion, extract_citations, extract_message_content

_SEARCH_MODEL = "deepseek/deepseek-v4-flash-0731"
_MAX_RESULTS = 5


async def web_search(query: str, max_results: int = _MAX_RESULTS) -> dict[str, Any]:
    """Ищет информацию в интернете по запросу ученика.

    Args:
        query: Поисковый запрос на естественном языке.
        max_results: Сколько результатов запросить у поискового движка.

    Returns:
        Словарь с полями: `status`, `summary` (краткий ответ модели),
        `sources` (список `{title, url, snippet}`) либо `message` при ошибке.
    """
    query = (query or "").strip()
    if not query:
        return {"status": "error", "message": "Пустой поисковый запрос."}

    try:
        data = await chat_completion(
            model=_SEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — поисковый помощник репетитора. Найди актуальную и "
                        "достоверную информацию и кратко перескажи её на русском, "
                        "приоритет — учебные и проверяемые источники."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=800,
            plugins=[
                {
                    "id": "web",
                    "max_results": max(1, min(int(max_results), 10)),
                    "search_prompt": "Ищи актуальные и достоверные источники",
                }
            ],
        )
    except RouterAIError as exc:
        return {"status": "error", "message": str(exc)}

    sources = [
        {
            "title": cit.get("title") or "",
            "url": cit.get("url") or "",
            "snippet": (cit.get("content") or "")[:300],
        }
        for cit in extract_citations(data)
    ]
    return {
        "status": "success",
        "summary": extract_message_content(data),
        "sources": sources,
    }
