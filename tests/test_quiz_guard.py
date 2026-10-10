"""Тесты серверной валидации квиза (`quiz_guard`)."""

from __future__ import annotations

from tutor_agent.schemas import AnswerType, Difficulty, QuizCard
from tutor_agent.tools.quiz_guard import (
    build_regen_instruction,
    difficulty_of,
    validate_quiz,
)


def _card(**kwargs) -> QuizCard:
    base = dict(
        question="Сколько будет 2+2?",
        answer_type=AnswerType.SINGLE,
        options=["3", "4", "5", "6"],
        correct_answer="4",
        difficulty=Difficulty.EASY,
    )
    base.update(kwargs)
    return QuizCard(**base)


def test_valid_single_quiz_passes() -> None:
    result = validate_quiz(_card())
    assert result.ok is True
    assert result.reasons == []


def test_not_a_question_is_rejected() -> None:
    result = validate_quiz(_card(question="Два плюс два равно четыре."))
    assert result.ok is False
    assert any("Вопрос" in r for r in result.reasons)


def test_single_requires_at_least_two_options() -> None:
    result = validate_quiz(_card(options=["4"]))
    assert result.ok is False
    assert any("варианта" in r for r in result.reasons)


def test_duplicate_options_are_rejected() -> None:
    result = validate_quiz(_card(options=["4", "4", "5", "6"]))
    assert result.ok is False
    assert any("дубли" in r for r in result.reasons)


def test_correct_answer_must_be_among_options() -> None:
    result = validate_quiz(_card(options=["3", "5", "6", "7"]))
    assert result.ok is False
    assert any("отсутствует" in r for r in result.reasons)


def test_answer_leak_in_question_is_rejected() -> None:
    # Ответ "четыре" прямо упомянут в вопросе.
    result = validate_quiz(
        _card(
            question="Верно ли, что ответ четыре на вопрос 2+2?",
            options=["четыре", "пять"],
            correct_answer="четыре",
        )
    )
    assert result.ok is False
    assert any("раскрыт" in r for r in result.reasons)


def test_open_question_must_have_empty_options() -> None:
    result = validate_quiz(
        _card(answer_type=AnswerType.OPEN, options=["лишнее"], correct_answer="4")
    )
    assert result.ok is False
    assert any("open" in r for r in result.reasons)


def test_open_question_requires_answer() -> None:
    result = validate_quiz(
        _card(answer_type=AnswerType.OPEN, options=[], correct_answer="   ")
    )
    assert result.ok is False
    assert any("эталон" in r.lower() or "Пустой" in r for r in result.reasons)


def test_build_regen_instruction_lists_reasons() -> None:
    text = build_regen_instruction(["дубли вариантов", "нет вопроса"])
    assert "дубли вариантов" in text
    assert "нет вопроса" in text
    assert "JSON" in text


def test_difficulty_of_normalizes() -> None:
    assert difficulty_of("hard") is Difficulty.HARD
    assert difficulty_of(Difficulty.EASY) is Difficulty.EASY
    assert difficulty_of("нечто") is Difficulty.MEDIUM
