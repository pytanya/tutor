"""Корневой ADK-агент репетитора.

Модель — RouterAI через `LiteLlm` (см. [`tutor_agent.llm.routerai`](tutor_agent/llm/routerai.py)).
Инструменты: веб-поиск, выжимка урока, генерация квиза, чтение загруженного файла.

Обратите внимание: `output_schema` НЕ задаётся, так как агент использует
инструменты (не все не-Gemini модели поддерживают одновременный structured
output и tools). Модель инструктируется возвращать единый JSON-конверт текстом.

Конверт затем перехватывается [`render_envelope_after_model`](tutor_agent/callbacks.py:120):
он подменяет ответ модели на человекочитаемый текст (его показывает `adk web`),
а структурные данные кладёт в `actions.stateDelta` — их читает наш фронтенд.
"""

from __future__ import annotations

from google.adk.agents import Agent

from .callbacks import render_envelope_after_model
from .llm.routerai import ModelRole, get_model
from .prompts import ENVELOPE_FORMAT, TUTOR_PERSONA
from .tools import generate_quiz, generate_summary, read_artifact, web_search

root_agent = Agent(
    name="tutor",
    model=get_model(ModelRole.PLANNER),
    description=(
        "Персональный репетитор для школьника: объясняет темы, проверяет "
        "понимание вопросами, делает выжимки, ищет информацию и читает файлы."
    ),
    instruction=TUTOR_PERSONA + "\n\n" + ENVELOPE_FORMAT,
    after_model_callback=render_envelope_after_model,
    tools=[
        web_search,
        generate_summary,
        generate_quiz,
        read_artifact,
    ],
)
