"""Инструмент чтения загруженного учеником файла.

Файлы хранятся как ADK-артефакты сессии: загрузка идёт встроенным эндпоинтом
`POST /apps/{app}/users/{user}/sessions/{session}/artifacts` (его использует наш
UI). Инструмент достаёт содержимое артефакта и возвращает текст, пригодный для
передачи модели в контекст.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from ..config import get_settings

# MIME-типы и расширения, которые читаем как обычный текст.
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/csv",
}
_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".xml")


def _is_text_like(filename: str, mime_type: str) -> bool:
    mime = (mime_type or "").lower()
    if mime.startswith(_TEXT_MIME_PREFIXES) or mime in _TEXT_MIME_EXACT:
        return True
    return filename.lower().endswith(_TEXT_EXTENSIONS)


async def read_artifact(filename: str, tool_context: ToolContext) -> dict[str, Any]:
    """Читает содержимое загруженного файла (артефакта сессии).

    Args:
        filename: Имя загруженного файла.
        tool_context: Контекст инструмента (предоставляется ADK автоматически).

    Returns:
        Словарь с полями `status`, `filename`, `text` (для текстовых файлов)
        либо `message` с пояснением.
    """
    settings = get_settings()

    available = await tool_context.list_artifacts()
    if not available:
        return {
            "status": "error",
            "message": "Ученик ещё не загрузил ни одного файла.",
        }
    if filename not in available:
        return {
            "status": "error",
            "message": f"Файл «{filename}» не найден. Доступные файлы: {available}.",
        }

    part = await tool_context.load_artifact(filename=filename)
    if part is None:
        return {"status": "error", "message": f"Не удалось прочитать файл «{filename}»."}

    # Часть может быть простым текстом.
    if getattr(part, "text", None):
        return {
            "status": "success",
            "filename": filename,
            "text": part.text[: settings.tool_max_context_chars],
        }

    inline = getattr(part, "inline_data", None)
    if inline is not None:
        mime_type = getattr(inline, "mime_type", "") or ""
        raw: bytes = getattr(inline, "data", b"") or b""
        if _is_text_like(filename, mime_type):
            text = raw.decode("utf-8", errors="replace")
            return {
                "status": "success",
                "filename": filename,
                "text": text[: settings.tool_max_context_chars],
            }
        return {
            "status": "error",
            "message": (
                f"Формат файла «{filename}» ({mime_type or 'неизвестен'}) не "
                "поддерживается для чтения в MVP. Поддерживаются текстовые "
                "форматы: txt, md, csv, json, yaml, xml."
            ),
        }

    return {"status": "error", "message": f"Пустое содержимое файла «{filename}»."}
