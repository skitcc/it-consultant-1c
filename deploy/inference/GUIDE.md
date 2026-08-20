# Инструкция: диагностика Ollama и переход на vLLM

Документ для себя через неделю. Скрипты лежат в `deploy/inference/`. Код приложения на сервере 1. GPU и Ollama/vLLM — на сервере 2.

Два разных `.env`:

- Этот каталог, сервер 2: `deploy/inference/.env` — Docker, HF-кэш, порты, доля VRAM.
- Приложение, сервер 1: `/etc/it-consultant/.env` или корневой `.env` — `INFERENCE_BACKEND`, URL моделей.

Их нельзя смешивать. `RERANK_MODEL` в приложении — имя модели для клиента. В `deploy/inference/.env` то же имя — Hugging Face id для контейнера.

---

## Карта команд

Все команды из `deploy/inference/` (на сервере 2, если не сказано иное).

```bash
cd deploy/inference
cp .env.example .env    # один раз
```

- Первый заход на GPU: `./scripts/ollama_diag.sh` — что загружено, VRAM, `OLLAMA_*`.
- Понять, выгружается ли LLM: `./scripts/ollama_pipeline_trace.sh` — `load_duration` по шагам письма.
- То же, как в проде: `./scripts/ollama_pipeline_trace.sh --full` — 20 rerank, как `RAG_CANDIDATES`.
- Смотреть пилу VRAM: `./scripts/gpu_watch.sh -- ./scripts/ollama_pipeline_trace.sh`.
- Влезают ли 4 модели сразу: `./scripts/ollama_preload.sh`.
- Скачать веса HF: `./scripts/hf_download.sh` (~95+ ГБ в `HF_HOME`).
- Проверить нарезку GPU: `./scripts/vram_budget.sh` (сумма util ≤ 0.90).
- Поднять vLLM: `./scripts/vllm_up.sh`.
- Живы ли порты: `./scripts/vllm_status.sh`.
- Логи: `./scripts/vllm_logs.sh llm` (имена `llm|rerank|vlm|embed`).
- Остановить vLLM: `./scripts/vllm_down.sh`.
- Замер Ollama: `./scripts/bench_pipeline.sh --backend ollama --full`.
- Замер vLLM: `./scripts/bench_pipeline.sh --backend vllm --full --compare out/bench_ollama.json`.
- Нужен ли reindex: `./scripts/compare_embeddings.sh`.

JSON пишется в `deploy/inference/out/`.

С ноутбука или сервера 1 (только HTTP, без `nvidia-smi`):

```bash
OLLAMA_BASE_URL=http://server2:11434 ./scripts/ollama_diag.sh
```

---

## 1. Диагностика Ollama

Цель: отличить «модель каждый раз грузится в GPU» от «долго думает».

### 1.1. `ollama_diag.sh`

Что делает: `/api/version`, `/api/tags`, `/api/ps`, `OLLAMA_*` из окружения, `nvidia-smi` если есть.

Как читать `/api/ps`:

- 0 моделей — холодный старт, первый запрос всегда с load.
- 1–2 модели — на письме не хватит (нужны embed + rerank + LLM).
- 4 модели и `expires_at` с нормальной датой — `keep_alive` не вечный, через 5 минут выгрузят.
- `processor` / мало `size_vram` относительно `size` — часть слоёв на CPU, будет медленно без unload.
- `OLLAMA_MAX_LOADED_MODELS` пусто или 1–3 — четвёртая роль вытеснит одну из трёх. Это главная гипотеза.

`nvidia-smi`: если used ≈ total, четыре модели вместе не живут. Запиши used для `gpt-oss:120b`, когда она одна в памяти — это ориентир для `LLM_GPU_UTIL`.

### 1.2. `ollama_pipeline_trace.sh`

Имитирует письмо: embed → N rerank chat → LLM `think=medium`.

Колонки:

- `wall` — всё время шага.
- `load` — загрузка весов в GPU. Ollama отдаёт наносекунды, скрипт переводит в секунды.
- `prefill` — разбор промпта.
- `eval` — генерация токенов (у LLM сюда входит thinking).

Правила:

1. У шага `llm` после embed/rerank `load` ≥ 2 с — веса сняли и подняли снова. Это вытеснение, не «модель тупит».
2. `load` у LLM ≈ 0, а `eval` десятки секунд — thinking. vLLM тут поможет меньше, чем батч-rerank.
3. У `rerank_1/N` большой `load`, у `rerank_2` почти 0 — reranker прогрелся, но в проде всё равно 20 последовательных chat.
4. Большой `load` на каждом rerank — runner падает (OOM) или меняется `num_ctx`.

`--full` = 20 кандидатов. Без флага = 3 (дым, быстрее).

### 1.3. `gpu_watch.sh`

Только сервер 2.

```bash
./scripts/gpu_watch.sh
./scripts/gpu_watch.sh -- ./scripts/ollama_pipeline_trace.sh
```

Пила used-memory вверх-вниз на каждом шаге = модели сменяют друг друга. Полка на одном уровне = резидентные веса.

### 1.4. `ollama_preload.sh`

Грузит embed, rerank, VLM, LLM с `keep_alive=-1`. Если в `/api/ps` меньше четырёх — вместе не живут. Это не лечение прода, только факт.

---

## 2. Модели Hugging Face

Ollama-тег `gpt-oss:120b` не заменяем другой сетью. HF-id: [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (MoE, MXFP4, чекпоинт ~61 ГБ, runtime примерно 63–80 ГБ). vLLM это умеет официально. `think=medium` → `reasoning_effort=medium`.

Соответствия:

- LLM: `gpt-oss:120b` → `openai/gpt-oss-120b`
- Rerank: `dengcao/Qwen3-Reranker-8B:Q8_0` → `Qwen/Qwen3-Reranker-8B`
- VLM: `qwen3-vl:8b` → `Qwen/Qwen3-VL-8B-Instruct` (не Thinking)
- Embed: `nomic-embed-text` → `nomic-ai/nomic-embed-text-v1.5`

Запасной LLM, если 120B + трое не влезают: `openai/gpt-oss-20b` (~16 ГБ). Меняется `LLM_MODEL` в `.env`, не код.

### Скачивание

На сервере 2 с интернетом:

```bash
pip install -U huggingface_hub
# при необходимости: huggingface-cli login
export HF_HOME=/opt/hf-cache
./scripts/hf_download.sh
```

Кэш должен совпадать с `HF_HOME` в `.env` — тот же путь монтируется в контейнеры как `/root/.cache/huggingface`. Повторный `vllm_up` не качает заново, если файлы уже там.

Офлайн: `hf download ...` на машине с сетью, затем:

```bash
rsync -aP /opt/hf-cache/ user@server2:/opt/hf-cache/
```

Диск: свободно не меньше 120 ГБ (веса + docker layers + запас). gpt-oss обычно без гейта. Если 401 — токен read на huggingface.co.

Не путать GGUF из Ollama (`~/.ollama/models`) с HF-кэшем. vLLM GGUF не ест.

---

## 3. Поднять vLLM

Четыре процесса на одной карте. Не один контейнер на все роли.

Перед стартом:

```bash
./scripts/vram_budget.sh
```

Сумма `gpu_memory_utilization` должна быть ≤ 0.90. Флаг считается от всей карты, не от свободной памяти. Дефолты в `models.yaml` / `.env.example` подобраны под ~120 ГБ и `max-model-len=8192`. Если поднять `LLM_MAX_MODEL_LEN` до 32k–128k, KV-кэш съест карту — не делай этого.

```bash
./scripts/vllm_up.sh
./scripts/vllm_status.sh
./scripts/vllm_logs.sh llm
```

Первый старт LLM — минуты (чтение ~61 ГБ). Если контейнер рестартится:

1. OOM — уменьши `LLM_GPU_UTIL` (например 0.50), `./scripts/vllm_up.sh vllm-llm`.
2. `--reasoning-parser openai_gptoss` неизвестен — в логе будет допустимое имя; поправь `compose.yml` или `LLM_EXTRA_ARGS`.
3. rerank 400 «does not support Score API» — не трогай `hf_overrides` в compose.

Поменять параметр на живом сервере:

1. Правка `deploy/inference/.env` (`LLM_GPU_UTIL=0.50`).
2. `./scripts/vram_budget.sh`
3. `./scripts/vllm_up.sh` — compose пересоздаст изменённые сервисы.

Остановить: `./scripts/vllm_down.sh`. Ollama при этом не трогается.

Порты: 8001 LLM, 8002 rerank, 8003 VLM, 8004 embed.

Другой проект бьёт в `http://server2:8001/v1`. Чтобы добавить пятую модель: блок в `models.yaml`, сервис в `compose.yml`, переменные в `.env.example`, свободный порт.

---

## 4. Бенчмарки

Сначала база, пока Ollama ещё жив:

```bash
./scripts/bench_pipeline.sh --backend ollama --full
```

Файл: `out/bench_ollama.json`.

Холодный Ollama (сначала выгрузить всех):

```bash
./scripts/bench_pipeline.sh --backend ollama --full --evict
```

После `vllm_up`:

```bash
./scripts/bench_pipeline.sh --backend vllm --full --warm \
  --compare out/bench_ollama.json
```

Смотри:

- `rerank_http_calls`: 20 у Ollama, 1 у vLLM. Если wall rerank не упал на порядок — vLLM rerank не прогрелся или не тот endpoint.
- `load_sec` у vLLM нет / ~0: веса резидентны. Если wall LLM всё ещё огромный — это generation, не load.
- `--warm` отбрасывает первый прогон (компиляция CUDA).

---

## 5. Переключить приложение (сервер 1)

Код с `INFERENCE_BACKEND=ollama` (дефолт) ходит в Ollama как раньше: `/api/chat`, `/api/embed`, `keep_alive=-1`, `think`. Новые URL игнорируются.

Проверка после деплоя кода, ещё на Ollama: письма ходят, в логе `backend=ollama`.

Потом в `.env` приложения:

```bash
INFERENCE_BACKEND=vllm
OLLAMA_MODEL=gpt-oss:120b
LLM_BASE_URL=http://server2:8001/v1
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_BASE_URL=http://server2:8004/v1
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
RERANK_BASE_URL=http://server2:8002/v1
RERANK_MODEL=Qwen/Qwen3-Reranker-8B
VLM_BASE_URL=http://server2:8003/v1
VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

Рестарт `mail-gateway` и `reindex`. В логе: `backend=vllm`.

Откат: `INFERENCE_BACKEND=ollama` и старые `OLLAMA_MODEL` / `EMBEDDING_MODEL` / `RERANK_MODEL` / `VLM_MODEL`. Ollama не удалять, пока не пройдут живые письма.

Если при старте `EMBEDDING_BASE_URL is required when INFERENCE_BACKEND=vllm` — не все нужные URL заданы.

### Embeddings и Qdrant

После переключения embedder:

```bash
./scripts/compare_embeddings.sh
```

Нужны живые Ollama и vLLM-embed одновременно.

- cosine > 0.99 и одинаковая размерность (768) — индекс можно не трогать.
- < 0.90 или разный dim — `python -m reindex --once`. Не делать это молча, если cosine уже 1.0.

Rerank на vLLM — логиты yes/no, не «напечатай 0.73». Порог `RAG_SCORE_THRESHOLD` после перехода пересмотри на паре писем.

---

## 6. Порядок на неделе работ

1. Диагностика Ollama (`diag` + `pipeline_trace` + при возможности `gpu_watch`). Сохрани JSON из `out/`.
2. `hf_download.sh`, `vram_budget.sh`, `vllm_up.sh`. Ollama пока не глушить.
3. `bench_pipeline` ollama vs vllm, `compare_embeddings`.
4. Деплой кода, сначала `INFERENCE_BACKEND=ollama`.
5. Переключить на `vllm`, 1–2 тестовых письма, при необходимости reindex.
6. Когда стабильно — можно выключить Ollama. Откат = одна переменная.

---

## 7. Типичные поломки

- Письмо 2–5 мин, в trace LLM `load` большой — вытеснение. Пока на Ollama костыль: `OLLAMA_MAX_LOADED_MODELS=4` и preload. Цель — vLLM.
- Письмо долго, LLM `load≈0`, `eval` большой — thinking. Уменьшать max tokens / effort.
- vLLM LLM не стартует, CUDA OOM — `LLM_GPU_UTIL` вниз, не поднимать `MAX_MODEL_LEN`.
- Второй контейнер падает при старте — сумма util больше свободной памяти. Стартовать по одному: `vllm_up.sh vllm-llm`.
- Rerank 400 — проверь `hf_overrides` и `--runner pooling`.
- Ответ с `<think>` в письме — нет `--reasoning-parser` у LLM.
- RAG «поехал» — `compare_embeddings`; если плохо — reindex.
- Factory упал на старте — забыл URL при `INFERENCE_BACKEND=vllm`.
- `nvidia-smi not found` — скрипт не на GPU-сервере. HTTP-диагностика всё равно работает.

Контекст gpt-oss в карточке модели — до 128k. В нашем стеке держим 8192, как `OLLAMA_CONTEXT_LENGTH`. Иначе одна LLM займёт всю карту.
