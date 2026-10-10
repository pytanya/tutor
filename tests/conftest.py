"""Общие фикстуры тестов."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tutor_agent.tools import quiz


@pytest.fixture(autouse=True)
def _reset_quiz_streak() -> None:
    """Сбрасывает модульный счётчик отказов квиза перед каждым тестом."""
    quiz._reject_streak = 0


def chat_response(content: str) -> dict[str, Any]:
    """Собирает ответ RouterAI в формате Chat Completion с заданным текстом."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def chat_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Ответ RouterAI, где content — это JSON-строка."""
    return chat_response(json.dumps(payload, ensure_ascii=False))
