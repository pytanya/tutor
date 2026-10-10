"""Схемы данных репетитора.

Переносим из `pytanya/edututor` идею единого «конверта» ответа: любой ответ
ученику — строгий JSON `{type, text, payload, difficulty}`. В ADK схема
используется как `output_schema` агента, чтобы модель возвращала валидный
структурированный результат.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Конверт ответа
# ---------------------------------------------------------------------------


class ContentType(str, Enum):
    """Тип учебного контента в ответе."""

    THEORY = "theory"
    PRACTICE = "practice"
    HINT = "hint"
    QUIZ = "quiz"
    EVALUATION = "evaluation"
    SUMMARY = "summary"


class Difficulty(str, Enum):
    """Сложность материала."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContentEnvelope(BaseModel):
    """Единый конверт ответа репетитора ученику."""

    type: ContentType = Field(description="Тип контента.")
    text: str = Field(description="Основной текст ответа для ученика.")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Доп. данные (например, visuals, карточка квиза).",
    )
    difficulty: Difficulty = Field(
        default=Difficulty.MEDIUM, description="Сложность материала."
    )


# ---------------------------------------------------------------------------
# Карточка квиза
# ---------------------------------------------------------------------------


class AnswerType(str, Enum):
    """Тип ответа в вопросе квиза."""

    SINGLE = "single"  # один вариант из списка
    OPEN = "open"      # открытый ответ


class QuizCard(BaseModel):
    """Карточка проверочного вопроса.

    Поле `correct_answer` НИКОГДА не отдаётся ученику — оно используется только
    на сервере для проверки и остаётся в `payload` под приватным ключом.
    """

    question: str = Field(description="Прямой вопрос с «?» без правильного ответа.")
    answer_type: AnswerType = Field(description="Тип ответа: single или open.")
    options: list[str] = Field(
        default_factory=list,
        description="Варианты для single (ровно 4: 1 верный + 3 дистрактора); "
        "пусто для open.",
    )
    correct_answer: str = Field(
        description="Эталонный ответ. НЕ показывать ученику."
    )
    difficulty: Difficulty = Field(description="Сложность вопроса.")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        return value.strip()

    def to_student_payload(self) -> dict[str, Any]:
        """Возвращает payload для ученика БЕЗ правильного ответа."""
        return {
            "question": self.question,
            "answer_type": self.answer_type.value,
            "options": list(self.options),
            "difficulty": self.difficulty.value,
        }


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_FIRST_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_payload(raw: str) -> dict[str, Any]:
    """Устойчиво извлекает JSON-объект из «сырого» ответа модели.

    Снимает markdown-обёртку ```json ... ``` и лишний текст вокруг объекта —
    тот же приём, что `_parse_quiz_card` в edututor.

    Args:
        raw: Текст ответа модели.

    Returns:
        Разобранный словарь.

    Raises:
        ValueError: Если JSON-объект не найден или невалиден.
    """
    text = _FENCE_RE.sub("", raw or "").strip()
    match = _FIRST_JSON_RE.search(text)
    if not match:
        raise ValueError("В ответе модели не найден JSON-объект.")
    return json.loads(match.group(0))
