"""Тесты тонкого лончера: агент обнаруживается, UI раздаётся.

Проверяем без обращения к сети и без запуска модели: сам факт сборки
приложения и наличие маршрутов.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_agent_is_discovered() -> None:
    response = _client().get("/list-apps")
    assert response.status_code == 200
    assert "tutor_agent" in response.json()


def test_ui_is_served_on_tutor_mount() -> None:
    response = _client().get("/tutor/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "тетрадь" in response.text


def test_ui_contains_voice_and_upload_controls() -> None:
    html = _client().get("/tutor/").text
    # Ключевые client-side возможности присутствуют в разметке/скрипте.
    assert "speechSynthesis" in html          # озвучка (TTS)
    assert "SpeechRecognition" in html        # надиктовка (STT)
    assert "/run_sse" in html                 # стриминг чата
    assert "/artifacts" in html               # загрузка файла
