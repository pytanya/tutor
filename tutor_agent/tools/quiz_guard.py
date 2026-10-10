"""Серверная валидация сгенерированного квиза (порт идеи `quiz_guard`).

Модель генерирует квиз свободным текстом, поэтому возможны дефекты:
вопрос-утверждение с уже названным ответом, дубли вариантов, эталон не из
вариантов и т.п. Здесь проверяем карточку до показа ученику и возвращаем
причины отказа, которые можно передать в «тихую» регенерацию.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import AnswerType, Difficulty, QuizCard

_QUESTION_MARK_RE = re.compile(r"\?")
_QUESTION_WORDS = (
    "что", "кто", "как", "почему", "зачем", "где", "когда", "какой", "какая",
    "какое", "какие", "чем", "сколько", "верно ли", "правда ли", "выбери",
    "выберите", "найди", "определи", "чему",
)
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass
class GuardResult:
    """Результат проверки карточки квиза."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def _looks_like_question(text: str) -> bool:
    low = text.lower()
    if _QUESTION_MARK_RE.search(text):
        return True
    return any(word in low for word in _QUESTION_WORDS)


def _leaks_answer(question: str, correct_answer: str) -> bool:
    """Проверяет, не назван ли правильный ответ прямо в тексте вопроса."""
    answer = _normalize(correct_answer)
    if not answer:
        return False
    question_norm = _normalize(question)
    # Прямое вхождение нормализованного ответа в вопрос.
    if answer and answer in question_norm:
        return True
    # Совпадение по значимым словам ответа (для многословных эталонов).
    answer_words = {w for w in answer.split() if len(w) > 3}
    if answer_words:
        question_words = set(question_norm.split())
        if answer_words.issubset(question_words):
            return True
    return False


def _normalize(text: str) -> str:
    """Приводит строку к нижнему регистру без пунктуации и лишних пробелов."""
    return " ".join(_WORD_RE.sub(" ", (text or "").lower()).split())


def validate_quiz(card: QuizCard) -> GuardResult:
    """Проверяет карточку квиза на структурные и смысловые дефекты.

    Args:
        card: Карточка квиза.

    Returns:
        `GuardResult` с признаком валидности и списком причин отказа.
    """
    reasons: list[str] = []

    # 1. Вопросность текста.
    if not _looks_like_question(card.question):
        reasons.append("Вопрос не сформулирован как вопрос (нет «?» или вопросительного слова).")

    # 2. Структура по типу ответа.
    if card.answer_type is AnswerType.SINGLE:
        if len(card.options) < 2:
            reasons.append("Для single-вопроса нужно минимум 2 варианта.")
        normalized_options = [_normalize(opt) for opt in card.options]
        if len(set(normalized_options)) != len(normalized_options):
            reasons.append("Варианты ответа содержат дубли.")
        if _normalize(card.correct_answer) not in set(normalized_options):
            reasons.append("Правильный ответ отсутствует среди вариантов.")
    else:  # open
        if card.options:
            reasons.append("Для open-вопроса список вариантов должен быть пустым.")
        if not card.correct_answer.strip():
            reasons.append("Для open-вопроса нужен эталонный ответ.")

    # 3. Утечка ответа в тексте вопроса.
    if _leaks_answer(card.question, card.correct_answer):
        reasons.append("Правильный ответ раскрыт в тексте вопроса.")

    # 4. Эталон не должен быть пустым.
    if not card.correct_answer.strip():
        reasons.append("Пустой эталонный ответ.")

    return GuardResult(ok=not reasons, reasons=reasons)


def build_regen_instruction(reasons: list[str]) -> str:
    """Строит промпт «тихой» регенерации — только из причин, без утечки ответа.

    Args:
        reasons: Причины отказа предыдущей карточки.

    Returns:
        Инструкция для повторной генерации.
    """
    bullets = "\n".join(f"- {reason}" for reason in reasons)
    return (
        "Предыдущий вопрос был отклонён. Исправь перечисленные проблемы и "
        "верни ТОЛЬКО новый JSON-объект вопроса в том же формате:\n" + bullets
    )


def difficulty_of(value: str | Difficulty) -> Difficulty:
    """Нормализует значение сложности к `Difficulty`."""
    if isinstance(value, Difficulty):
        return value
    try:
        return Difficulty(value)
    except ValueError:
        return Difficulty.MEDIUM
