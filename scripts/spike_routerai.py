"""Спайк-скрипт: проверка подключения RouterAI.

Проверяет три вещи:
1. Есть ли ключ `ROUTERAI_API_KEY` в окружении/`.env`.
2. Работает ли прямой вызов `/chat/completions` (используется инструментами).
3. Работает ли ADK `LiteLlm` с RouterAI (используется агентом) — тот путь,
   на котором строятся все диалоги.

Запуск:
    python -m scripts.spike_routerai
или (если скрипт запускается как файл):
    python scripts/spike_routerai.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Позволяем запускать как файл: добавляем корень проекта в sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tutor_agent.config import get_settings  # noqa: E402
from tutor_agent.llm.client import RouterAIError, chat_completion, extract_message_content  # noqa: E402


async def _check_direct(settings) -> bool:
    """Прямой вызов Chat Completions через `llm.client`."""
    print("\n[2/3] Прямой вызов /chat/completions ...", flush=True)
    try:
        data = await chat_completion(
            model=settings.tutor_model_fast,
            messages=[
                {"role": "user", "content": "Ответь одним словом: работает?"},
            ],
            max_tokens=20,
            temperature=0.0,
        )
    except RouterAIError as exc:
        print(f"   ОШИБКА: {exc}")
        return False
    print(f"   OK. Ответ модели: {extract_message_content(data)!r}")
    return True


async def _check_adk_litellm(settings) -> bool:
    """Вызов модели через ADK `LiteLlm` (путь агента)."""
    print("\n[3/3] Вызов через ADK LiteLlm ...", flush=True)
    try:
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types

        from tutor_agent.llm.routerai import ModelRole, get_model

        model = get_model(ModelRole.FAST)
        request = LlmRequest(
            model=settings.tutor_model_fast,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text="Скажи «работает» одним словом.")],
                )
            ],
        )
        text = ""
        async for response in model.generate_content_async(request):
            if response.content and response.content.parts:
                for part in response.content.parts:
                    if getattr(part, "text", None):
                        text += part.text
        print(f"   OK. Ответ LiteLlm: {text.strip()!r}")
        return bool(text.strip())
    except Exception as exc:  # noqa: BLE001 — спайк: показываем любую ошибку
        print(f"   ОШИБКА: {type(exc).__name__}: {exc}")
        return False


async def main() -> int:
    settings = get_settings()

    print("[1/3] Проверка ключа ROUTERAI_API_KEY ...", flush=True)
    key = settings.routerai_api_key or ""
    if not key:
        print(
            "   КЛЮЧ НЕ ЗАДАН. Скопируйте .env.example в .env и укажите\n"
            "   ROUTERAI_API_KEY (https://routerai.ru/settings/keys)."
        )
        return 2
    masked = key[:6] + "…" + key[-4:] if len(key) > 12 else "***"
    print(f"   OK. Ключ: {masked}; base_url: {settings.routerai_base_url}")
    print(
        "   Модели: planner="
        f"{settings.tutor_model_planner}, fast={settings.tutor_model_fast}, "
        f"judge={settings.tutor_model_judge}"
    )

    direct_ok = await _check_direct(settings)
    adk_ok = await _check_adk_litellm(settings)

    print("\nИтог:")
    print(f"   прямой /chat/completions: {'OK' if direct_ok else 'ОШИБКА'}")
    print(f"   ADK LiteLlm:              {'OK' if adk_ok else 'ОШИБКА'}")
    return 0 if (direct_ok and adk_ok) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
