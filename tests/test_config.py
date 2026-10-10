"""Тесты конфигурации и фабрики моделей RouterAI."""

from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from tutor_agent.llm.routerai import ModelRole, get_model


def test_get_model_returns_litellm_with_routerai_model() -> None:
    model = get_model(ModelRole.PLANNER)
    assert isinstance(model, LiteLlm)
    # id модели пробрасывается в litellm как есть (без двойного префикса).
    assert model.model == "deepseek/deepseek-v3.2"


def test_conversational_roles_are_not_reasoning_models() -> None:
    """planner/fast не должны быть «рассуждающими» моделями.

    У таких моделей ответ уходит в канал `reasoning`, а `content` остаётся
    пустым — ADK завершает ход ошибкой MODEL_RETURNED_NO_CONTENT.
    """
    forbidden = ("deepseek-v4-flash", "qwen3-30b-a3b", "qwen3.5-9b")
    for role in (ModelRole.PLANNER, ModelRole.FAST):
        model_id = get_model(role).model
        assert not any(bad in model_id for bad in forbidden), model_id


def test_conversational_roles_avoid_openai_prefix() -> None:
    """Префикс `openai/` на пути LiteLlm срезается → RouterAI не находит модель.

    Работают только vendor-префиксные id (deepseek/…, qwen/…, google/…,
    meta-llama/…).
    """
    for role in (ModelRole.PLANNER, ModelRole.FAST):
        model_id = get_model(role).model
        assert not model_id.startswith("openai/"), model_id


def test_all_roles_build() -> None:
    for role in ModelRole:
        model = get_model(role)
        assert isinstance(model, LiteLlm)
        assert model.model  # непустой id
