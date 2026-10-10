# Репетитор для школьника

AI-репетитор на **Google ADK** с моделями **RouterAI**. Интерфейс чата,
наблюдаемость, сессии и хранилище файлов берутся из встроенного ADK
(`adk web`), а голос, генерация выжимки и проверочных вопросов добавлены
сверху.

## Возможности

- **Чат с репетитором** — объяснения по шагам, на русском языке.
- **Голос (без серверных аудио-моделей)** — надиктовка и озвучка ответов
  средствами браузера: Web Speech API (`SpeechRecognition` и `speechSynthesis`).
- **Проверочные вопросы (квиз)** — отдельный инструмент `generate_quiz` со
  строгим JSON-форматом, серверной валидацией (`quiz_guard`) и «тихой»
  регенерацией. Правильный ответ никогда не попадает ученику.
- **Выжимка урока** — короткое резюме через быструю/дешёвую модель
  (`generate_summary`).
- **Веб-поиск** — через плагин веб-поиска RouterAI (`web_search`).
- **Загрузка файла** — как ADK-артефакт сессии; инструмент `read_artifact`
  подгружает текст в контекст (txt, md, csv, json, yaml, xml).
- **Наблюдаемость** — встроенные трейсы и dev-UI ADK.

Все модели идут через один OpenAI-совместимый эндпоинт RouterAI
(`https://routerai.ru/api/v1`); Google API key не нужен.

## Требования

- Python 3.10+ (проверено на 3.14).
- Ключ RouterAI: https://routerai.ru/settings/keys
- Браузер с поддержкой Web Speech API для голоса (**Chrome/Edge**;
  в Firefox распознавание может отсутствовать — есть текстовый ввод).

## Установка

```bat
python -m pip install -r requirements.txt
```

## Настройка

Скопируйте образец и заполните ключ:

```bat
copy .env.example .env
```

В `.env` обязателен только `ROUTERAI_API_KEY`. Модели задаются переменными
`TUTOR_MODEL_PLANNER`, `TUTOR_MODEL_FAST`, `TUTOR_MODEL_JUDGE`.

При выборе моделей RouterAI есть два подводных камня:

> ⚠️ **1. Не ставьте «рассуждающие» (reasoning) модели на `planner`/`fast`.**
> Такие модели (`deepseek/deepseek-v4-flash-0731`, `qwen/qwen3-30b-a3b`,
> `qwen/qwen3.5-9b`) отдают цепочку размышлений в отдельный канал `reasoning`,
> а поле `content` может остаться пустым. ADK видит только thought-части и
> завершает ход ошибкой `MODEL_RETURNED_NO_CONTENT`.

> ⚠️ **2. Не используйте id с префиксом `openai/…`** (например
> `openai/gpt-4o-mini`). На пути ADK `LiteLlm` с
> `custom_llm_provider="openai"` litellm срезает префикс `openai/` и отправляет
> «голое» имя, которого RouterAI не знает (`Model 'gpt-4o-mini' not found`).
> Работают только vendor-префиксные id: `deepseek/…`, `qwen/…`, `google/…`,
> `meta-llama/…`.

Проверенные модели для `planner`/`fast` (отдают `content`, поддерживают
tool-calling, идут через `LiteLlm`): `deepseek/deepseek-v3.2`,
`qwen/qwen-2.5-72b-instruct`, `google/gemini-2.5-flash`,
`meta-llama/llama-3.3-70b-instruct`.
Гибридные `qwen/qwen3.7-flash`/`qwen3.6-flash`/`qwen3.8-flash` тоже отдают
текст, но остаются reasoning-моделями (риск при длинном размышлении).

## Проверка подключения (спайк)

Убедиться, что ключ и модели работают (прямой вызов и путь через ADK `LiteLlm`):

```bat
python -m scripts.spike_routerai
```

## Диагностика моделей

Если в ответе приходит `MODEL_RETURNED_NO_CONTENT` (или пустой ответ), проверьте,
какие модели возвращают текст в `content`, а не только в `reasoning`:

```bat
python -m scripts.diag_models    :: content vs reasoning по списку моделей
python -m scripts.diag_tools     :: поддержка tool-calling
python -m scripts.diag_litellm   :: какие id проходят через путь ADK LiteLlm
python -m scripts.smoke_agent    :: полный прогон агента одним запросом
```

Если `diag_models` показывает `NO_CONTENT` — модель «рассуждающая», замените её.
Если `diag_litellm` падает с `Model 'X' not found` — уберите префикс `openai/`.
Замените `planner`/`fast` на модель из проверенного списка выше (например
`deepseek/deepseek-v3.2`).

## Запуск

**Вариант 1 — штатный ADK (чат + наблюдаемость):**

```bat
adk web .
```

Откройте http://127.0.0.1:8000 — встроенный dev-UI с чатом и трейсами.

**Вариант 2 — с нашим UI (голос, квиз, загрузка файла):**

```bat
python -m server.app
```

Откройте http://127.0.0.1:8080/tutor/ — тот же ADK-сервер, плюс наша
страница (голос, рендер квиза, загрузка файла) на том же origin.
Dev-UI ADK при этом доступен по http://127.0.0.1:8080/dev-ui/.

**Вариант 3 — диалог в терминале:**

```bat
adk run tutor_agent
```

> Если в консоли Windows «кракозябры» вместо русского текста, выполните
> `set PYTHONUTF8=1` (или `chcp 65001`) перед запуском.

## Как это устроено

Агент `tutor_agent` использует модель-планировщик RouterAI и набор
инструментов. Веб-слой не переписывает транспорт: `server/app.py` вызывает
встроенную фабрику ADK `get_fast_api_app(web=True)` (тот же app, что поднимает
`adk web`) и добавляет к нему статический UI из `frontend/` по `/tutor/`.

### Конверт ответа и красивый вывод

Модель отвечает строгим JSON-конвертом `ContentEnvelope`
(`{type, text, payload, difficulty}`) — это контракт контента. Чтобы в
интерфейсах был виден человекочитаемый текст, а не сырой JSON, у агента
задан `after_model_callback`
[`render_envelope_after_model`](tutor_agent/callbacks.py:120):

- ответ модели подменяется на markdown-текст из поля `text` конверта — его
  показывают и встроенный `adk web`, и наш фронтенд;
- полный структурный конверт (в т.ч. `payload` квиза) кладётся в
  `actions.stateDelta` события под ключом `temp:tutor_envelope`; фронтенд
  читает его из SSE-потока (`/run_sse`) и отрисовывает интерактивные элементы
  (варианты ответа квиза). Префикс `temp:` не сохраняется в историю сессии.

```
tutor/
  tutor_agent/            # ADK-пакет (обнаруживается `adk web .`)
    agent.py              # root_agent
    callbacks.py          # after_model_callback: конверт → markdown + stateDelta
    config.py             # настройки и фабрика моделей (planner/fast/judge)
    schemas.py            # ContentEnvelope, QuizCard, парсинг JSON
    prompts.py            # персона и промпты (квиз, выжимка, конверт)
    llm/
      routerai.py         # фабрика LiteLlm
      client.py           # прямые вызовы RouterAI (chat/completions, plugins)
    tools/
      web_search.py       # веб-поиск через плагин RouterAI
      quiz.py             # generate_quiz + вызов quiz_guard
      quiz_guard.py       # серверная валидация квиза
      summary.py          # generate_summary
      artifacts.py        # read_artifact (чтение загруженного файла)
  server/
    app.py                # лончер: get_fast_api_app(web=True) + mount frontend/
  frontend/
    index.html            # чат (SSE) + Web Speech API + upload + рендер квиза
  scripts/
    spike_routerai.py     # проверка подключения RouterAI
    diag_models.py        # content vs reasoning по списку моделей
    diag_tools.py         # проверка поддержки tool-calling
    diag_litellm.py       # какие id проходят через путь ADK LiteLlm
    smoke_agent.py        # полный прогон агента одним запросом
  tests/                  # pytest
  .env.example
  requirements.txt
  plans/tutor-plan.md
```

## Тесты

```bat
python -m pytest
```

Покрыты: схемы и парсинг JSON, `quiz_guard`, инструменты (с замоканным
RouterAI), `after_model_callback` (конверт → markdown + структурный канал),
конфигурация моделей и связка сервера (обнаружение агента, раздача UI).

## Безопасность

- Ключи хранятся только в `.env` (в репозиторий не коммитится).
- Правильный ответ квиза не отдаётся ученику: он остаётся под приватным ключом
  на сервере.
- Встроенный сервер ADK не имеет аутентификации и предназначен для локального
  запуска. Не выставляйте его в публичную сеть; для локального доступа хост по
  умолчанию — `127.0.0.1`.

## Дорожная карта

- Учебная база (RAG) и разделение «учебные материалы vs интернет».
- Интервальное повторение (SM-2) и адаптивная сложность.
- Оценка открытых ответов моделью «Решения» RouterAI.
