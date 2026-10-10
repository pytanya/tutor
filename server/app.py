"""Тонкий лончер поверх встроенного ADK-сервера.

Не переписывает транспорт: вызывает штатную фабрику
`google.adk.cli.fast_api.get_fast_api_app(web=True)` — это ровно тот app,
который поднимает команда `adk web` (эндпоинты `/run`, `/run_sse`,
`/run_live`, sessions, artifacts, dev-UI и трейсы).

Дополнительно монтирует наш статический UI (`frontend/`) по `/tutor/`,
чтобы чат-фронтенд и API жили на одном origin (без CORS).

Запуск:
    python -m server.app
Эквивалент штатного запуска без нашего UI:
    adk web .
"""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.cli.fast_api import get_fast_api_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
AGENTS_DIR = str(PROJECT_ROOT)

APP_NAME = "tutor_agent"
UI_MOUNT_PATH = "/tutor"


def _session_service_uri() -> str | None:
    """URI сервиса сессий (например, sqlite:///sessions.db).

    Пустое значение — использовать in-memory (по умолчанию в ADK).
    """
    value = os.getenv("TUTOR_SESSION_SERVICE_URI", "").strip()
    return value or None


def _allow_origins() -> list[str] | None:
    """Дополнительные CORS-origins через запятую (для отдельного фронтенда)."""
    raw = os.getenv("TUTOR_ALLOW_ORIGINS", "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def create_app():
    """Собирает FastAPI-app: встроенный ADK (web=True) + наш статический UI."""
    app = get_fast_api_app(
        agents_dir=AGENTS_DIR,
        web=True,
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8080")),
        allow_origins=_allow_origins(),
        session_service_uri=_session_service_uri(),
    )

    if FRONTEND_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            UI_MOUNT_PATH,
            StaticFiles(directory=str(FRONTEND_DIR), html=True),
            name="tutor-ui",
        )
    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8080"))
    reload = os.getenv("APP_RELOAD", "1").strip() not in {"0", "false", "False"}
    print(f"Tutor UI:      http://{host}:{port}{UI_MOUNT_PATH}/")
    print(f"ADK dev-UI:    http://{host}:{port}/dev-ui/")
    print(f"App name:      {APP_NAME}")
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(PROJECT_ROOT / "tutor_agent"), str(FRONTEND_DIR)],
    )


if __name__ == "__main__":
    main()
