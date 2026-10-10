"""Тесты инструментов репетитора с замоканным RouterAI (без сети)."""

from __future__ import annotations

from typing import Any

import importlib

import pytest

from tests.conftest import chat_response, chat_response_json
from tutor_agent.llm import client as client_module

# В пакете `tutor_agent.tools` имена `web_search`/`summary`/`quiz`
# реэкспортированы как функции и перекрывают одноимённые подмодули, поэтому
# достаём сами модули через `importlib` (возвращает реальный модуль из sys.modules).
quiz_module = importlib.import_module("tutor_agent.tools.quiz")
summary_module = importlib.import_module("tutor_agent.tools.summary")
web_search_module = importlib.import_module("tutor_agent.tools.web_search")


def _patch_chat(monkeypatch, monkey_target, response: dict[str, Any]) -> dict[str, Any]:
    """Подменяет `chat_completion` и запоминает переданные аргументы."""
    captured: dict[str, Any] = {}

    async def fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(monkey_target, "chat_completion", fake_chat_completion)
    return captured


# ── web_search ────────────────────────────────────────────────────────────


async def test_web_search_success(monkeypatch) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "Краткий ответ",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "title": "Источник",
                                "url": "https://example.org",
                                "content": "Фрагмент текста",
                            },
                        }
                    ],
                }
            }
        ]
    }
    captured = _patch_chat(monkeypatch, web_search_module, response)

    result = await web_search_module.web_search("что такое фотосинтез", max_results=3)

    assert result["status"] == "success"
    assert result["summary"] == "Краткий ответ"
    assert result["sources"][0]["url"] == "https://example.org"
    # Проверяем, что передан плагин веб-поиска.
    assert captured["plugins"][0]["id"] == "web"
    assert captured["plugins"][0]["max_results"] == 3


async def test_web_search_empty_query() -> None:
    result = await web_search_module.web_search("   ")
    assert result["status"] == "error"


# ── generate_summary ──────────────────────────────────────────────────────


async def test_generate_summary_success(monkeypatch) -> None:
    captured = _patch_chat(
        monkeypatch, summary_module, chat_response("Это выжимка урока.")
    )

    result = await summary_module.generate_summary("Очень длинный текст урока...")

    assert result == {"status": "success", "summary": "Это выжимка урока."}
    assert captured["max_tokens"] == 400


async def test_generate_summary_empty_text() -> None:
    result = await summary_module.generate_summary("  ")
    assert result["status"] == "error"


# ── generate_quiz ─────────────────────────────────────────────────────────


def _valid_quiz_payload() -> dict[str, Any]:
    return {
        "question": "Чему равен квадрат числа 5?",
        "answer_type": "single",
        "options": ["10", "25", "20", "15"],
        "correct_answer": "25",
        "difficulty": "easy",
    }


async def test_generate_quiz_success_hides_answer(monkeypatch) -> None:
    _patch_chat(monkeypatch, quiz_module, chat_response_json(_valid_quiz_payload()))

    result = await quiz_module.generate_quiz("квадраты чисел")

    assert result["status"] == "success"
    assert result["type"] == "quiz"
    assert result["question"] == "Чему равен квадрат числа 5?"
    assert result["options"] == ["10", "25", "20", "15"]
    # Правильный ответ — только под приватным ключом.
    assert result["_correct_answer"] == "25"
    assert "correct_answer" not in result


async def test_generate_quiz_regenerates_on_bad_card(monkeypatch) -> None:
    """Первая карточка с дублями → регенерация → валидная карточка."""
    calls = {"n": 0}

    async def fake_chat_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            bad = {
                "question": "Выбери верное: 2+2?",
                "answer_type": "single",
                "options": ["4", "4", "5"],
                "correct_answer": "4",
                "difficulty": "easy",
            }
            return chat_response_json(bad)
        return chat_response_json(_valid_quiz_payload())

    monkeypatch.setattr(quiz_module, "chat_completion", fake_chat_completion)

    result = await quiz_module.generate_quiz("арифметика")

    assert result["status"] == "success"
    assert calls["n"] == 2


async def test_generate_quiz_blocks_after_reject_cap(monkeypatch) -> None:
    """После серии отказов квиз блокируется (аналог quiz_reject_streak)."""

    async def always_bad(**kwargs):
        bad = {
            "question": "Утверждение без вопроса.",
            "answer_type": "single",
            "options": ["1", "1"],
            "correct_answer": "9",
            "difficulty": "easy",
        }
        return chat_response_json(bad)

    monkeypatch.setattr(quiz_module, "chat_completion", always_bad)

    first = await quiz_module.generate_quiz("тема")
    assert first["status"] == "error"

    second = await quiz_module.generate_quiz("тема")
    assert second["status"] == "error"

    third = await quiz_module.generate_quiz("тема")
    assert third["status"] == "blocked"


async def test_generate_quiz_empty_topic() -> None:
    result = await quiz_module.generate_quiz("  ")
    assert result["status"] == "error"


# ── llm.client ────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    last_payload: dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, json=None, headers=None):  # noqa: A002
        _FakeAsyncClient.last_payload = {"url": url, "json": json, "headers": headers}
        return _FakeResponse(200, {"choices": [{"message": {"content": "ок"}}]})


async def test_chat_completion_builds_request(monkeypatch) -> None:
    monkeypatch.setenv("ROUTERAI_API_KEY", "test-key")
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")
    client_module.get_settings.cache_clear()
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

    try:
        data = await client_module.chat_completion(
            model="deepseek/deepseek-v3.2",
            messages=[{"role": "user", "content": "привет"}],
            max_tokens=123,
        )
    finally:
        client_module.get_settings.cache_clear()

    assert data["choices"][0]["message"]["content"] == "ок"
    sent = _FakeAsyncClient.last_payload
    assert sent["url"] == "https://routerai.ru/api/v1/chat/completions"
    assert sent["json"]["max_tokens"] == 123
    assert sent["headers"]["Authorization"] == "Bearer test-key"


# ── extract_message_content: reasoning-модели ──────────────────────────────


def test_extract_message_content_returns_text() -> None:
    data = {"choices": [{"message": {"content": "  ответ  "}}]}
    assert client_module.extract_message_content(data) == "  ответ  "


def test_extract_message_content_empty_content_is_empty_string() -> None:
    data = {"choices": [{"message": {"content": None}}]}
    assert client_module.extract_message_content(data) == ""


def test_extract_message_content_reasoning_only_raises() -> None:
    """Только reasoning без content (рассуждающая модель) → понятная ошибка."""
    data = {"choices": [{"message": {"content": None, "reasoning": "думаю..."}}]}
    with pytest.raises(client_module.RouterAIError):
        client_module.extract_message_content(data)


def test_extract_message_content_reasoning_content_only_raises() -> None:
    data = {
        "choices": [{"message": {"content": "", "reasoning_content": "думаю..."}}]
    }
    with pytest.raises(client_module.RouterAIError):
        client_module.extract_message_content(data)


async def test_chat_completion_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ROUTERAI_API_KEY", raising=False)
    client_module.get_settings.cache_clear()
    try:
        with pytest.raises(client_module.RouterAIError):
            await client_module.chat_completion(model="m", messages=[])
    finally:
        client_module.get_settings.cache_clear()
