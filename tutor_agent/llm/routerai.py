"""Фабрика моделей RouterAI для Google ADK.

ADK-класс [`LiteLlm`](tutor_agent/llm/routerai.py) принимает `model` и
произвольные kwargs, которые без изменений уходят в вызов litellm. Поэтому
`api_base` и `api_key` передаются напрямую — задавать переменные окружения
`OPENAI_API_BASE` не требуется.

Имя модели RouterAI содержит собственный слэш-префикс провайдера
(например `deepseek/deepseek-v4-flash-0731`). Чтобы litellm не пытался
интерпретировать его как свой провайдер, указывается явный
`custom_llm_provider="openai"`, а базовый URL задаётся через `api_base`.
"""

from __future__ import annotations

from enum import Enum

from google.adk.models.lite_llm import LiteLlm

from ..config import get_settings


class ModelRole(str, Enum):
    """Роли моделей (перенос идеи planner/fast/judge из edututor)."""

    PLANNER = "planner"  # ведение диалога, объяснения
    FAST = "fast"        # выжимка урока, черновики
    JUDGE = "judge"      # оценка ответов, маршрутизация (модель «Решения»)


def _model_id(role: ModelRole) -> str:
    settings = get_settings()
    return {
        ModelRole.PLANNER: settings.tutor_model_planner,
        ModelRole.FAST: settings.tutor_model_fast,
        ModelRole.JUDGE: settings.tutor_model_judge,
    }[role]


def get_model(role: ModelRole = ModelRole.PLANNER) -> LiteLlm:
    """Возвращает настроенную ADK-модель `LiteLlm`, указывающую на RouterAI.

    Args:
        role: Роль модели (`planner` | `fast` | `judge`).

    Returns:
        Экземпляр `LiteLlm`, готовый к использованию в `LlmAgent(model=...)`.
    """
    settings = get_settings()
    return LiteLlm(
        model=_model_id(role),
        custom_llm_provider="openai",     # трактуем как OpenAI-совместимый API
        api_base=settings.routerai_base_url,
        api_key=settings.routerai_api_key,
        drop_params=True,                 # выкидывать неподдерживаемые параметры
    )
