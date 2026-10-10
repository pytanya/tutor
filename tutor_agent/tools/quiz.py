"""Инструмент генерации проверочного вопроса (квиза).

Отдельный LLM-вызов (как `generate_quiz` в edututor) со строгим JSON-форматом,
последующей серверной валидацией (`quiz_guard`) и ограниченной «тихой»
регенерацией. Правильный ответ никогда не попадает в результат для ученика —
он остаётся под приватным ключом `_correct_answer`.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..config import get_settings
from ..llm.client import RouterAIError, chat_completion, extract_message_content
from ..prompts import QUIZ_SYSTEM_PROMPT
from ..schemas import (
    AnswerType,
    ContentType,
    Difficulty,
    QuizCard,
    parse_json_payload,
)
from .quiz_guard import build_regen_instruction, difficulty_of, validate_quiz

# Счётчик последовательных отказов на уровне процесса (аналог quiz_reject_streak).
_reject_streak = 0


def _build_card(data: dict[str, Any], difficulty: Difficulty) -> QuizCard:
    """Собирает `QuizCard` из «сырого» JSON модели, устойчиво к вариациям ключей."""
    answer_type_raw = str(data.get("answer_type", "single")).strip().lower()
    answer_type = (
        AnswerType.OPEN if answer_type_raw == "open" else AnswerType.SINGLE
    )
    options = data.get("options") or []
    if not isinstance(options, list):
        options = []
    return QuizCard(
        question=str(data.get("question", "")).strip(),
        answer_type=answer_type,
        options=[str(opt) for opt in options],
        correct_answer=str(data.get("correct_answer", "")).strip(),
        difficulty=difficulty,
    )


async def generate_quiz(
    topic: str, difficulty: str = "medium", context: str = ""
) -> dict[str, Any]:
    """Генерирует проверочный вопрос по теме.

    Args:
        topic: Тема вопроса.
        difficulty: Сложность — `easy` | `medium` | `hard`.
        context: Дополнительный контекст (материал урока, факты).

    Returns:
        При успехе — словарь с карточкой БЕЗ правильного ответа
        (`status`, `type`, `question`, `answer_type`, `options`, `difficulty`).
        При ошибке — `{"status": "error", "message": ...}`; если квиз
        заблокирован после серии отказов — `{"status": "blocked", ...}`.
    """
    global _reject_streak

    settings = get_settings()
    topic = (topic or "").strip()
    if not topic:
        return {"status": "error", "message": "Не указана тема для квиза."}

    diff = difficulty_of(difficulty)
    if _reject_streak >= settings.quiz_reject_cap:
        return {
            "status": "blocked",
            "message": (
                "Генерация квиза временно недоступна из-за серии ошибок — "
                "предложи ученику практическое задание."
            ),
        }

    context_snippet = (context or "").strip()[: settings.tool_max_context_chars]
    base_user = f"Тема: {topic}\nСложность: {diff.value}"
    if context_snippet:
        base_user += f"\n\nФакты из материала:\n{context_snippet}"

    user_prompt = base_user
    last_reasons: list[str] = []

    # Одна попытка + до `quiz_reject_cap` регенераций.
    for _attempt in range(settings.quiz_reject_cap + 1):
        try:
            data = await chat_completion(
                model=settings.tutor_model_fast,
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=settings.quiz_max_tokens,
                temperature=0.7,
            )
            payload = parse_json_payload(extract_message_content(data))
            card = _build_card(payload, diff)
        except (RouterAIError, ValueError, ValidationError) as exc:
            last_reasons = [f"Некорректный ответ модели: {exc}"]
            user_prompt = base_user + "\n\n" + build_regen_instruction(last_reasons)
            continue

        result = validate_quiz(card)
        if result.ok:
            _reject_streak = 0
            return {
                "status": "success",
                "type": ContentType.QUIZ.value,
                "difficulty": card.difficulty.value,
                **card.to_student_payload(),
                # Приватно — для серверной проверки, ученику не показывать.
                "_correct_answer": card.correct_answer,
            }

        last_reasons = result.reasons
        user_prompt = base_user + "\n\n" + build_regen_instruction(last_reasons)

    _reject_streak += 1
    return {
        "status": "error",
        "message": "Не удалось сгенерировать корректный вопрос.",
        "reasons": last_reasons,
    }
