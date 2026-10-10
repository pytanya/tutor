"""Тесты схем: конверт ответа, карточка квиза, парсинг JSON."""

from __future__ import annotations

import pytest

from tutor_agent.schemas import (
    AnswerType,
    ContentEnvelope,
    ContentType,
    Difficulty,
    QuizCard,
    parse_json_payload,
)


def test_content_envelope_defaults() -> None:
    env = ContentEnvelope(type=ContentType.THEORY, text="Объяснение")
    assert env.difficulty is Difficulty.MEDIUM
    assert env.payload == {}


def test_content_envelope_accepts_type_from_string() -> None:
    env = ContentEnvelope(type="quiz", text="?", payload={"question": "2+2?"})
    assert env.type is ContentType.QUIZ


def test_quiz_card_student_payload_hides_answer() -> None:
    card = QuizCard(
        question="Сколько будет 2+2?",
        answer_type=AnswerType.SINGLE,
        options=["3", "4", "5", "6"],
        correct_answer="4",
        difficulty=Difficulty.EASY,
    )
    payload = card.to_student_payload()
    assert payload["question"] == "Сколько будет 2+2?"
    assert payload["options"] == ["3", "4", "5", "6"]
    assert "correct_answer" not in payload
    assert "4" not in payload.values()


def test_quiz_card_strips_question() -> None:
    card = QuizCard(
        question="  Вопрос?  ",
        answer_type=AnswerType.OPEN,
        options=[],
        correct_answer="да",
        difficulty=Difficulty.MEDIUM,
    )
    assert card.question == "Вопрос?"


def test_parse_json_payload_plain() -> None:
    data = parse_json_payload('{"a": 1, "b": "текст"}')
    assert data == {"a": 1, "b": "текст"}


def test_parse_json_payload_strips_markdown_fence() -> None:
    raw = '```json\n{"question": "2+2?", "options": ["3", "4"]}\n```'
    data = parse_json_payload(raw)
    assert data["question"] == "2+2?"


def test_parse_json_payload_extracts_object_from_prose() -> None:
    raw = 'Вот вопрос: {"question": "2+2?", "correct_answer": "4"} — готово.'
    data = parse_json_payload(raw)
    assert data["correct_answer"] == "4"


def test_parse_json_payload_raises_without_json() -> None:
    with pytest.raises(ValueError):
        parse_json_payload("здесь нет никакого объекта")
