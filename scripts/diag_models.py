"""Диагностика моделей RouterAI: проверка, возвращают ли они текст в `content`.

Причина ошибки MODEL_RETURNED_NO_CONTENT — «рассуждающие» модели отдают
chain-of-thought в поле `reasoning` и оставляют `content` пустым. ADK трактует
такой ответ как thought-only и помечает turn ошибкой.

Запуск:
    python -m scripts.diag_models
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tutor_agent.config import get_settings  # noqa: E402
from tutor_agent.llm.client import RouterAIError, chat_completion  # noqa: E402

CANDIDATES = [
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-latest",
]

PROMPT = 'Ответь ровно одной строкой JSON: {"ok": true}'


async def probe(model: str) -> None:
    try:
        data = await chat_completion(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=120,
            temperature=0.0,
        )
    except RouterAIError as exc:
        print(f"{model:42s} ERROR: {str(exc)[:90]}")
        return

    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    finish = choice.get("finish_reason")
    usage = data.get("usage", {}) or {}
    r_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    verdict = "CONTENT_OK" if (isinstance(content, str) and content.strip()) else "NO_CONTENT"
    print(
        f"{model:42s} {verdict:10s} finish={finish!s:8s} "
        f"reasoning_tokens={r_tokens!s:5s} content={content!r:40s} "
        f"has_reasoning={bool(reasoning)}"
    )


async def main() -> int:
    get_settings()  # прогреваем конфиг (и .env)
    print(f"Проверяю {len(CANDIDATES)} моделей RouterAI...\n")
    for model in CANDIDATES:
        await probe(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
