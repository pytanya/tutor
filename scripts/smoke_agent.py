"""Дымовой тест агента-репетитора через ADK Runner.

Прогоняет один реальный запрос ученика и печатает финальный ответ + проверку,
не возникла ли ошибка MODEL_RETURNED_NO_CONTENT (пустой ответ).

Запуск:
    python -m scripts.smoke_agent
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from tutor_agent.agent import root_agent  # noqa: E402

PROMPT = "Привет! Помоги подготовиться к уроку по биологии, тема «Высшие споровые растения»."
APP = "tutor_agent"


async def main() -> int:
    runner = InMemoryRunner(agent=root_agent, app_name=APP)
    session = await runner.session_service.create_session(
        app_name=APP, user_id="smoke-user"
    )

    message = types.Content(role="user", parts=[types.Part(text=PROMPT)])

    final_text = ""
    error_code = None
    async for event in runner.run_async(
        user_id="smoke-user", session_id=session.id, new_message=message
    ):
        if ev_error := getattr(event, "error_code", None):
            error_code = ev_error
        content = getattr(event, "content", None)
        if content and content.parts:
            for part in content.parts:
                text = getattr(part, "text", None)
                if text and not getattr(part, "thought", False):
                    final_text += text

    print("=" * 70)
    print(f"error_code: {error_code!r}")
    print(f"длина ответа: {len(final_text)} символов")
    print("-" * 70)
    print(final_text[:1200] or "<ПУСТОЙ ОТВЕТ>")
    print("=" * 70)

    if error_code or not final_text.strip():
        print("\nРЕЗУЛЬТАТ: ПРОВАЛ — агент не вернул содержательный ответ.")
        return 1
    print("\nРЕЗУЛЬТАТ: OK — агент вернул ответ без ошибок.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
