# План: Репетитор для школьника на Google ADK + RouterAI

## 1. Цель и требования

Построить AI-репетитора для школьника на базе **Google ADK** (Python), полностью на моделях и сервисах **RouterAI** (без Google API key).

Требования (сопоставление с реальностью ADK):

| Требование | Что даёт ADK | Решение |
|---|---|---|
| Интерфейс чата | `adk web` (dev-UI :8000), `adk api_server` (`/run`, `/run_sse`) | Используем как есть |
| Надиктовывание (voice) | Только Gemini Live API (`*-live-*`) — **недоступно на RouterAI** | **Встроенные средства браузера (Web Speech API)**: `SpeechRecognition` (STT, надиктовка) и `speechSynthesis` (TTS, озвучка). Без серверных аудио-эндпоинтов и без аудио-моделей RouterAI |
| Поисковые движки | `google_search` — **только Gemini** | Свой tool `web_search` через плагин веб-поиска RouterAI (`:online` / `id:web`) + fallback на внешний провайдер |
| Загрузка файла | `ArtifactService` + CRUD артефактов в API-сервере (готового UI-виджета нет) | Свой эндпоинт загрузки → ADK-артефакт → tool чтения артефакта в контексте |
| Генерация выжимки/квиза/вопросов | Нет — берём идеи из edututor | Отдельные tools `generate_summary`, `generate_quiz` + серверный `quiz_guard` |

## 2. Что переносим из `pytanya/edututor`

1. **Строгий JSON-конверт ответа** `{type, text, payload, difficulty}` как контракт контента (в ADK — `output_schema` + `output_key`).
2. **Отдельный tool `generate_quiz`** + серверный `quiz_guard`: структурная валидация, детект утечки `_correct_answer`, «тихая» регенерация, блокировка quiz после N неудач.
3. **Выжимка урока** — отдельный дешёвый LLM-вызов (2–4 предложения, без markdown).
4. **Роли моделей planner/fast/judge** — на RouterAI-моделях; дорогая модель для объяснений, дешёвая для выжимки/квиза.
5. **Разделение поиска**: учебная база (RAG) vs интернет (web_search), с circuit breaker и лимитами.
6. **Опционально**: SM-2 повторение, адаптивная сложность (LinUCB) — как расширение после MVP.

## 3. Архитектура

```mermaid
flowchart TB
    subgraph FE [Frontend (frontend/index.html)]
      Chat[Чат + стриминг SSE]
      PTT[Push-to-talk: Web Speech API]
      TTS[speechSynthesis озвучка]
      Up[Загрузка файла]
    end
    subgraph SRV [Встроенный ADK-сервер]
      DevUI[adk web dev-UI + observability]
      RUN[POST /run_sse]
      ART[POST .../artifacts (upload)]
      SESS[GET/POST sessions]
    end
    subgraph ADK [ADK Agent Package]
      Root[tutor_agent root]
      Ex[explain_agent]
      Qz[quiz_agent]
      Sm[summary_agent]
      T1[tool web_search]
      T2[tool rag_search]
      T3[tool generate_quiz]
      T4[tool generate_summary]
      T5[tool read_artifact]
      G[quiz_guard]
    end
    subgraph RAI [RouterAI]
      LLM[chat models]
      DEC[decisions: Jev]
      WEB[web plugin]
    end
    Chat --> RUN --> Root
    PTT -->|текст| Chat
    TTS --> Chat
    Up --> ART
    SESS --> Chat
    DevUI -.наблюдаемость.-> RUN
    Root --> Ex
    Root --> Qz
    Root --> Sm
    Ex --> T1
    Ex --> T2
    Qz --> T3 --> G
    Sm --> T4
    Root --> T5
    T1 --> WEB
    T2 --> RAI
    Root --> LLM
    LLM --> RAI
```

### 3.1. Слои

- **ADK agent package** `tutor_agent/` — ядро. Загружается встроенными командами ADK: `adk web <agents_dir>` (чат + observability + CRUD артефактов) и `adk run` (CLI). Никакого собственного кода/эндпоинтов для чата не требуется.
- **Тонкий лончер** `server/app.py` — **не переписывает** транспорт, а вызывает встроенную фабрику ADK `google.adk.cli.fast_api.get_fast_api_app(agents_dir=..., web=True)`. Это тот же FastAPI-app, что поднимает `adk web`, поэтому доступны `/run`, `/run_sse`, `/run_live`, sessions, artifacts, dev-UI и трейсы. На тот же app монтируется наш статический UI (`frontend/`) по `/tutor/`, чтобы чат-фронтенд был same-origin с API (без CORS и без отдельного сервера). Лончер нужен только ради удобной одной команды запуска и монтирования кастомного UI под тем же origin.
- **Frontend** `frontend/index.html` — самостоятельный HTML/JS: чат через `POST /run_sse` (SSE), выбор/создание сессии через `/apps/{app}/users/{user}/sessions`, загрузка файла через `POST .../artifacts`. **Голос полностью на стороне браузера**: Web Speech API (`SpeechRecognition`/`webkitSpeechRecognition` для надиктовки, `speechSynthesis` для озвучки ответа). Серверных аудио-эндпоинтов нет.

### 3.2. Роутинг моделей (RouterAI)

Всё через `LiteLlm(model="openai/<routerai-model>")`, `OPENAI_API_BASE=https://routerai.ru/api/v1`, `OPENAI_API_KEY=<routerai key>`.

| Роль | Назначение | Модель (проверено, см. `scripts/diag_*`) |
|---|---|---|
| `planner` / объяснения | root-агент, сложные объяснения | `deepseek/deepseek-v3.2` (tools, без reasoning) |
| `fast` | выжимка, черновик квиза | `deepseek/deepseek-v3.2` |
| `judge` / решения | оценка ответов, маршрутизация | модель «Решения» `~typesafe/jev-latest` (`/api/v1/decisions`) |

> Все имена — id из каталога RouterAI. **Нельзя** ставить на `planner`/`fast` «рассуждающие» модели (`deepseek-v4-flash-0731`, `qwen3-30b-a3b`, `qwen3.5-9b`): их ответ уходит в канал `reasoning`, `content` пуст → ADK даёт `MODEL_RETURNED_NO_CONTENT`. Также нельзя id с префиксом `openai/` (litellm на пути `LiteLlm` срезает префикс и RouterAI не находит модель). Проверенные варианты: `deepseek/deepseek-v3.2`, `qwen/qwen-2.5-72b-instruct`, `google/gemini-2.5-flash`, `meta-llama/llama-3.3-70b-instruct`. Модели задаются через `.env` и легко заменяются.

### 3.3. Контракт контента

`ContentEnvelope` (Pydantic): `type ∈ {theory, practice, hint, quiz, evaluation, summary}`, `text: str`, `payload: dict` (в т.ч. `visuals`), `difficulty ∈ {easy, medium, hard}`. Устанавливается как `output_schema` соответствующего агента, кладётся в `output_key` и парсится сервером/frontend.

### 3.4. Квиз и контроль качества

- `generate_quiz(topic, difficulty, context_query)` — отдельный LLM-вызов, строгий JSON (`question`, `answer_type`, `options`, `_correct_answer`, `difficulty`).
- `quiz_guard`: структурная валидация (single → ≥2 опций, без дублей, `_correct_answer` ровно в `options`), проверка «вопросности», детект утечки ответа в тексте.
- Recovery: одна «тихая» регенерация (в промпт передаются только причины), затем `QUIZ_REJECT_CAP`; после — переключение на practice. `_correct_answer` никогда не отдаётся ученику.

### 3.5. Голос (push-to-talk, полностью в браузере)

1. Кнопка push-to-talk включает `SpeechRecognition` (Web Speech API) на языке `ru-RU`.
2. Распознанный текст подставляется в поле ввода и уходит в ADK-агента как обычное сообщение.
3. Ответ агента при включённой озвучке воспроизводится через `speechSynthesis` (`SpeechSynthesisUtterance`, `lang="ru-RU"`).
4. Никаких серверных STT/TTS и никаких аудио-моделей не используется. Требуется браузер с поддержкой Web Speech API (Chrome/Edge; в Firefox распознавание может отсутствовать — предусмотрен текстовый фолбэк).

### 3.6. Файлы

- Загрузка выполняется **встроенным** эндпоинтом артефактов: `POST /apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts` с телом `SaveArtifactRequest{filename, artifact: types.Part, custom_metadata}`. Файл сохраняется как ADK-артефакт сессии.
- Tool `read_artifact(filename)` подгружает содержимое в контекст (текст: txt/md/csv/json/yaml/xml). Для PDF — сообщаем о необходимости отдельного парсера (этап 2).
- MVP: поддержка текстовых форматов; загрузка — через встроенный API, свой эндпоинт не пишется.

## 4. Структура проекта

```
tutor/
  tutor_agent/            # ADK package (loadable by `adk web .`)
    __init__.py
    agent.py              # root_agent
    config.py             # настройки, фабрика моделей (роли planner/fast/judge)
    schemas.py            # ContentEnvelope, QuizCard
    prompts.py            # инструкции и промпты (quiz, summary, tutor persona)
    sub_agents/           # (этап 2: explain/quiz/summary как отдельные агенты)
    tools/
      web_search.py       # плагин веб-поиска RouterAI
      quiz.py             # generate_quiz + вызов quiz_guard
      quiz_guard.py       # серверная валидация квиза
      summary.py          # generate_summary
      artifacts.py        # read_artifact
    llm/
      routerai.py         # фабрика LiteLlm, роли моделей
      client.py           # прямые вызовы RouterAI (chat_completions, plugins)
  server/
    app.py                # тонкий лончер: get_fast_api_app(web=True) + mount frontend/ на /tutor
  frontend/
    index.html            # чат (SSE) + Web Speech API голос + upload + рендер квиза
  tests/
    test_quiz_guard.py
    test_schemas.py
    test_tools.py
  .env.example
  requirements.txt
  README.md
  plans/tutor-plan.md
```

### 4.1. Запуск

- **Штатный (рекомендуемый)**: `adk web .` (или `adk web tutor_agent`) — поднимает dev-UI с чатом и observability на `http://127.0.0.1:8000`. Наш UI — отдельно (`frontend/index.html`), при необходимости указывает `apiBase`.
- **Одна команда с нашим UI**: `python -m server.app` — поднимает тот же ADK FastAPI-app (`web=True`) и монтирует `frontend/` по `/tutor/`, поэтому чат-фронтенд и API живут на одном origin.
- **CLI**: `adk run tutor_agent` — быстрый диалог в терминале.

## 5. Риски и как закрываем

1. **`api_base` для RouterAI в `LiteLlm`** — способ (`api_base` vs env) проверяется **спайк-скриптом**. Рабочий вариант фиксируется в `config.py`.
2. **Двойной префикс** `openai/<vendor>/<model>` — проверяется тем же спайком (`custom_llm_provider="openai"`).
3. **Голос/поиск не Gemini** — голос на Web Speech API в браузере, поиск — плагин RouterAI. → Уже в архитектуре.
4. **`adk web` не содержит виджетов upload/voice** → голос клиентский (Web Speech API), upload — через встроенный artifacts-эндпоинт; свой чат-транспорт не пишем, используем `get_fast_api_app(web=True)`.
5. **Стоимость и лимиты** → бюджет-гард и лимиты токенов/размера результата по образцу edututor.

## 6. Этапы реализации

1. **Спайк интеграции**: подтвердить подключение RouterAI через `LiteLlm` (api_base, имя модели), `adk web` поднимается, простой агент отвечает.
2. **Каркас пакета**: `config.py` (роли моделей), `schemas.py`, базовый root-агент, `.env.example`, `requirements.txt`.
3. **Инструменты**: `web_search`, `generate_summary`, `generate_quiz` + `quiz_guard`, `read_artifact`.
4. **Сабагенты** (этап 2): explain / quiz / summary, роутинг через root.
5. **Тонкий лончер + Frontend MVP**: `server/app.py` (`get_fast_api_app(web=True)` + mount `frontend/`), HTML-чат с SSE (`/run_sse`), push-to-talk (Web Speech API), загрузка файла (artifacts API), рендер квиза.
6. **Тесты**: `quiz_guard`, схемы, tools (mock RouterAI).
7. **Документация и запуск**: README, `.env.example`, инструкции `adk web .`, `python -m server.app`.
8. **(Опционально, после MVP)**: RAG учебная база, SM-2 повторение, адаптивная сложность.

## 7. Критерии готовности MVP

- `adk web .` поднимает **встроенный** dev-UI (чат + observability), агент отвечает на вопросы школьника на русском.
- Наш `frontend/index.html` общается со встроенными эндпоинтами (`/run_sse`, sessions, artifacts) и запускается одной командой `python -m server.app`.
- Квиз генерируется валидным и не раскрывает правильный ответ; `quiz_guard` отсекает плохие карточки.
- Выжимка урока генерируется отдельным дешёвым вызовом.
- Веб-поиск работает через RouterAI-плагин.
- Файл загружается через встроенный artifacts-эндпоинт и его содержимое используется в ответе (`read_artifact`).
- Push-to-talk: голос (Web Speech API) → текст → ответ → озвучка (speechSynthesis).
- Есть тесты и README с инструкцией запуска; ключи только в `.env`.
