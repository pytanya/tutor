"""Конфигурация приложения и фабрика моделей RouterAI.

Все секреты и параметры читаются из окружения (`.env`). См. `.env.example`.

Ключевая идея: приложение не зависит от Gemini/Google API. Все модели
подключаются через OpenAI-совместимый эндпоинт RouterAI посредством ADK
`LiteLlm`, которому напрямую передаются `api_base` и `api_key`
(ADK пробрасывает произвольные kwargs в litellm).
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Загружаем .env один раз при импорте (значения в нём не перекрывают уже
# установленные переменные окружения по умолчанию dotenv).
load_dotenv()


class Settings(BaseSettings):
    """Настройки приложения, читаемые из окружения / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- RouterAI -----------------------------------------------------------
    routerai_api_key: str = ""
    routerai_base_url: str = "https://routerai.ru/api/v1"

    # --- Модели по ролям ----------------------------------------------------
    # Два ограничения при выборе моделей RouterAI:
    #
    # 1) НЕ берите «рассуждающие» (reasoning) модели на planner/fast. Такие
    #    модели (`deepseek/deepseek-v4-flash-0731`, `qwen/qwen3-30b-a3b`,
    #    `qwen/qwen3.5-9b`) отдают цепочку размышлений в канал `reasoning`, а
    #    поле `content` может остаться пустым. ADK видит только thought-части
    #    и завершает ход ошибкой MODEL_RETURNED_NO_CONTENT.
    #
    # 2) НЕ используйте id с префиксом `openai/...` (например
    #    `openai/gpt-4o-mini`): на пути ADK `LiteLlm` с
    #    `custom_llm_provider="openai"` litellm СРЕЗАЕТ префикс `openai/` и
    #    отправляет «голое» имя, которого RouterAI не знает
    #    (`Model 'gpt-4o-mini' not found`). Работают только vendor-префиксные
    #    id: `deepseek/...`, `qwen/...`, `google/...`, `meta-llama/...`.
    #
    # Проверенные модели (отдают `content`, поддерживают tool-calling, идут
    # через LiteLlm без срезания префикса):
    #   deepseek/deepseek-v3.2, qwen/qwen-2.5-72b-instruct,
    #   google/gemini-2.5-flash, meta-llama/llama-3.3-70b-instruct
    #
    # planner — ведение диалога и объяснения (tools, без reasoning)
    tutor_model_planner: str = "deepseek/deepseek-v3.2"
    # fast — выжимка урока, черновики, поиск (дёшево, без reasoning)
    tutor_model_fast: str = "deepseek/deepseek-v3.2"
    # judge — модель «Решения» для оценки ответов ученика и маршрутизации
    tutor_model_judge: str = "~typesafe/jev-latest"

    # --- Инструменты --------------------------------------------------------
    tool_max_context_chars: int = 4000
    quiz_max_tokens: int = 500
    quiz_reject_cap: int = 2

    # --- Приложение ---------------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    tutor_lang: str = "ru-RU"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает singleton-настройки (кэшируются)."""
    return Settings()
