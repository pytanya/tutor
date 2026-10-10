"""Инструмент генерации выжимки урока.

Перенос идеи из `pytanya/edututor`: короткое резюме через быструю/дешёвую
модель, с жёстким промптом (2–4 предложения, без markdown).
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..llm.client import RouterAIError, chat_completion, extract_message_content
from ..prompts import SUMMARY_SYSTEM_PROMPT


async def generate_summary(text: str) -> dict[str, Any]:
    """Делает краткую выжимку переданного учебного текста.

    Args:
        text: Текст урока/конспекта/материала для сжатия.

    Returns:
        Словарь вида `{"status": "success", "summary": "..."}` или `{"status":
        "error", "message": "..."}`.
    """
    settings = get_settings()
    text = (text or "").strip()
    if not text:
        return {"status": "error", "message": "Пустой текст для выжимки."}

    truncated = text[: settings.tool_max_context_chars]

    try:
        data = await chat_completion(
            model=settings.tutor_model_fast,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": truncated},
            ],
            max_tokens=400,
            temperature=0.3,
        )
    except RouterAIError as exc:
        return {"status": "error", "message": str(exc)}

    return {"status": "success", "summary": extract_message_content(data).strip()}
