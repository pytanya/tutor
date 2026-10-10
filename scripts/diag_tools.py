"""Диагностика: поддерживает ли модель RouterAI вызов инструментов (tools).

Агенту нужен tool-calling. Проверяем кандидатов на роль planner прямым
запросом `/chat/completions` с описанием функции.

Запуск:
    python -m scripts.diag_tools
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tutor_agent.config import get_settings  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": "Сгенерировать проверочный вопрос по теме.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Тема вопроса"},
                },
                "required": ["topic"],
            },
        },
    }
]

CANDIDATES = [
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]


async def probe(client: httpx.AsyncClient, base: str, key: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Составь проверочный вопрос про фотосинтез."}
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 300,
        "temperature": 0.0,
    }
    try:
        r = await client.post(
            f"{base.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    except httpx.HTTPError as exc:
        print(f"{model:40s} NETWORK ERROR: {exc}")
        return
    if r.status_code >= 400:
        print(f"{model:40s} HTTP {r.status_code}: {r.text[:80]}")
        return
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        print(f"{model:40s} TOOLS_OK  -> {fn.get('name')}({fn.get('arguments')!r})")
    else:
        content = (msg.get("content") or "")[:60]
        print(f"{model:40s} NO_TOOL_CALL  content={content!r}")


async def main() -> int:
    s = get_settings()
    print(f"Проверяю tool-calling на {len(CANDIDATES)} моделях...\n")
    async with httpx.AsyncClient(timeout=90.0) as client:
        for model in CANDIDATES:
            await probe(client, s.routerai_base_url, s.routerai_api_key, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
