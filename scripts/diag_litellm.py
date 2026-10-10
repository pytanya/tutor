"""Проверка моделей RouterAI через путь ADK LiteLlm (как у агента).

LiteLLM с `custom_llm_provider="openai"` по-разному обращается с префиксом
модели: для `openai/...` он срезает префикс и отправляет «голое» имя, которое
RouterAI не знает. Здесь проверяем, какие id реально принимаются этим путём.

Запуск:
    python -m scripts.diag_litellm
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tutor_agent.config import get_settings  # noqa: E402
from tutor_agent.llm.routerai import ModelRole  # noqa: E402

# Перебираем кандидатов через один и тот же LiteLlm-путь.
CANDIDATES = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.7-flash",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]


async def probe(model_id: str) -> None:
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    settings = get_settings()
    model = LiteLlm(
        model=model_id,
        custom_llm_provider="openai",
        api_base=settings.routerai_base_url,
        api_key=settings.routerai_api_key,
        drop_params=True,
    )
    request = LlmRequest(
        model=model_id,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="Ответь одним словом: работает?")],
            )
        ],
    )
    text = ""
    try:
        async for response in model.generate_content_async(request):
            if response.content and response.content.parts:
                for part in response.content.parts:
                    if getattr(part, "text", None) and not getattr(
                        part, "thought", False
                    ):
                        text += part.text
    except Exception as exc:  # noqa: BLE001
        print(f"{model_id:36s} ERROR: {type(exc).__name__}: {str(exc)[:110]}")
        return
    verdict = "OK" if text.strip() else "ПУСТО"
    print(f"{model_id:36s} {verdict:6s} -> {text.strip()[:50]!r}")


async def main() -> int:
    print(f"Проверяю {len(CANDIDATES)} моделей через LiteLlm-путь...\n")
    for model_id in CANDIDATES:
        await probe(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
