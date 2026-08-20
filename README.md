# IT Consultant

Репозиторий IT-консультанта: почтовый шлюз и сервис реиндексации файловой БД.

## Компоненты

| Пакет | Назначение |
|-------|------------|
| `common` | Общие настройки (`Settings`) и утилиты |
| `mail_gateway` | Exchange (EWS Streaming) → Qdrant RAG → Ollama или vLLM → reply |
| `reindex` | Следит за каталогом документов и индексирует их в Qdrant |
| `deploy/inference` | Диагностика Ollama, каталог vLLM, бенчмарки ([GUIDE.md](deploy/inference/GUIDE.md)) |

Общий конфиг — один класс [`common.Settings`](common/settings.py), читается из `.env` / переменных окружения. Оба сервиса используют одни и те же переменные. Инференс переключается `INFERENCE_BACKEND=ollama|vllm` (дефолт `ollama` — путь без изменений). Подробности и скрипты GPU-сервера: [`deploy/inference/GUIDE.md`](deploy/inference/GUIDE.md).

## Установка (разработка)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
# для reindex также (CPU torch, без CUDA-колёс):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[reindex,dev]"
cp .env.example .env
```

Заполните в `.env` поля `EWS_*` и `ADMIN_EMAIL` (и при необходимости `WATCH_PATH` / `DEBOUNCE_SECONDS`).

## Установка на сервер (systemd)

Скрипт [`deploy/install.sh`](deploy/install.sh) раскладывает дерево:

| Путь | Содержимое |
|------|------------|
| `/opt/it-consultant/.venv` | Python-окружение и пакеты |
| `/etc/it-consultant/.env` | секреты и настройки (из `.env.example`) |
| `/var/lib/it-consultant/db` | каталог файловой БД (`WATCH_PATH`) |
| `/etc/systemd/system/*.service` | unit-файлы + `it-consultant.target` |

```bash
# боевая установка (нужен root; создаёт user it-consultant и venv)
# на TTY спросит все переменные из .env.example (Enter — оставить текущее)
sudo ./deploy/install.sh --enable

# тот же install (код, venv, все зависимости .[reindex], все unit’ы),
# но enable/start только reindex (mail-gateway не запускается):
sudo ./deploy/install.sh --only reindex --enable

# только разложить файлы, без enable:
sudo ./deploy/install.sh
sudo systemctl daemon-reload
sudo systemctl enable --now it-consultant.target

# снять установку (stop/disable units, удалить app/env/data/units и user)
sudo ./deploy/install.sh --undeploy
```

`enable it-consultant.target` создаёт symlink’и в `multi-user.target.wants` и (через `Also=`) подтягивает `mail-gateway` и `reindex`. `--only reindex` (или `--only mail-gateway`) всё равно копирует приложение в `/opt/it-consultant` и ставит полный venv, но включает только выбранный unit.

При интерактивном запуске (stdin — TTY) скрипт парсит все `KEY=` из [`.env.example`](.env.example) (включая закомментированные опциональные) и спрашивает каждое значение. Пустой ввод (Enter) оставляет значение как в `.env` (или из `.env.example` при первой установке); для опциональных ключей, которых ещё нет в `.env`, Enter ничего не добавляет. Пароли/секреты (`*PASSWORD*`, `*SECRET*`, `*TOKEN*`) вводятся скрыто. Для CI / скриптов: `--no-configure` (или просто не-TTY — вопросы пропускаются); принудительно спросить даже без TTY: `--configure`.

`--undeploy` останавливает и отключает `it-consultant.target` (и связанные unit’ы), удаляет `/opt/it-consultant`, `/etc/it-consultant`, `/var/lib/it-consultant`, unit-файлы из `/etc/systemd/system/` и системного пользователя/группу `it-consultant`.

`install.sh` перед Docling ставит CPU-сборку PyTorch (`https://download.pytorch.org/whl/cpu`), без CUDA-колёс. На CPU-сервере пайплайн тот же, меняется только размер/тип wheel. Переопределение индекса: `TORCH_CPU_INDEX=...`.

### Безопасная проверка без трогания host rootfs

Никогда не гоняйте «тестовый» install в `/` без `--dest-dir`. Используйте фейковый root:

```bash
./deploy/install.sh --dest-dir /tmp/itc-root --layout-only
# дерево только под /tmp/itc-root/... ; systemctl и useradd не вызываются

./deploy/install.sh --dest-dir /tmp/itc-root   # + venv/pip внутрь /tmp/...

./deploy/install.sh --dest-dir /tmp/itc-root --undeploy  # снять только фейковое дерево

# неинтерактивно подставить первые переменные (остальные — Enter / EOF = defaults):
printf '%s\n' 'mail.example.com' 'bot@example.com' 'DOMAIN\bot' 'secret' \
  | ./deploy/install.sh --dest-dir /tmp/itc-root --layout-only --configure
```

Автотесты (тоже через `--dest-dir` во временный каталог):

```bash
pytest tests/deploy -q          # layout, undeploy, configure, содержимое unit-файлов
pytest tests/deploy -q -m slow  # полный install + smoke `python -m reindex` (нужен python3-venv)
```

Маркер `slow` пропускается, если на машине нельзя создать venv.

---

# Mail Gateway

Почтовый шлюз: Exchange (EWS Streaming) → RAG (Qdrant) → Ollama (`/api/chat`) → reply в тот же conversation.

## Структура

```
common/         # Settings, logging
mail_gateway/
  domain/       # модели (IncomingMessage, Reply)
  ports/        # контракты MailListener, MailSender, Assistant
  application/  # сценарий HandleIncomingMail
  adapters/     # реализации портов (EWS, Ollama, Qdrant RAG)
  main/         # composition root
tests/
```

Зависимости направлены внутрь: `adapters` → `ports`/`domain`, `application` → `ports`/`domain`.  
Адаптеры явно наследуют порты (`class EwsMailListener(MailListener)`).

## Поток

1. EWS Streaming: событие `NewMail` в Inbox.
2. Чтение письма → `conversation_id`, `item_id`, `change_key`, текст.
3. Загрузка треда, очистка тел.
4. Embedding вопроса → Qdrant `RAG_CANDIDATES` → rerank → `RAG_TOP_K` (соседи в том же разделе по `headings`) → фрагменты в `system_prompt`.
5. Non-stream вызов Ollama `/api/chat`: черновик с `think=medium`. Второй проход (`OLLAMA_VERIFIER_ENABLED`) по тем же чанкам с `think=high` по умолчанию выключен.
6. HTML-reply через EWS (таблицы, без Markdown и ссылок) + список использованных документов в конце.
7. Сбой, недоставленный ответ или несколько неотвеченных запросов пользователя в той же переписке — WARNING в лог и письмо на `ADMIN_EMAIL`.
8. При обрыве streaming — reconnect.

`change_key` — версия объекта письма в Exchange. Вместе с `item_id` однозначно указывает на конкретную ревизию письма; без него `get`/`reply` могут упасть, если письмо уже изменилось.

## Как протестировать самому

### 1. Unit-тесты (без Exchange)

```bash
pytest
```

Проверяют сценарий «письмо → ИИ → reply» на фейковых портах.

### 2. Round-trip с реальным ящиком + Ollama

1. Подними сервисы:
   ```bash
   docker compose up -d ollama qdrant
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. В `.env`:
   - `OLLAMA_BASE_URL=http://127.0.0.1:11434`
   - `OLLAMA_MODEL=llama3.2`
   - `EMBEDDING_MODEL=nomic-embed-text`
   - `QDRANT_URL=http://127.0.0.1:6333`
   - рабочие `EWS_*`
3. Проиндексируйте документы (`WATCH_PATH`) через `python -m reindex --once`.
4. Запуск шлюза:
   ```bash
   python -m mail_gateway
   ```
5. Напишите письмо на ящик бота. В логах будет `Assistant payload` (с `system_prompt` и фрагментами документации) и ответ модели.

## Контракт Ollama

Шлюз подтягивает тред из Exchange, чистит тела, достаёт кандидатов из Qdrant,
переранжирует их через Ollama `POST /api/chat` (`RERANK_MODEL` = Qwen3-Reranker:
Instruct/Query/Document → числовой score `0.00..1.00`), дополняет соседними чанками
и логирует внутренний payload. Для reasoning-модели `RERANK_NUM_PREDICT` включает
токены внутреннего thinking; если итоговый score не получен, запрос один раз
повторяется с отключённым thinking. При недоступности модели сохраняется fallback
на исходные vector scores.

Ответ пользователю строится черновиком одной `OLLAMA_MODEL`. Второй проход
по тем же чанкам (`OLLAMA_VERIFIER_ENABLED`) по умолчанию выключен: он правит
явные ошибки и не отбрасывает ответ из‑за цитат. В письмо уходит HTML без
reasoning, Markdown, URL и сносок `[1]`; в конец добавляются имена документов.

Параметры генерации: `OLLAMA_TEMPERATURE=0`, `OLLAMA_TOP_P=0.1`,
`OLLAMA_MAX_TOKENS=4096`, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_SEED=0`,
`OLLAMA_DRAFT_REASONING_EFFORT=medium`, `OLLAMA_VERIFIER_ENABLED=false`,
`OLLAMA_VERIFIER_REASONING_EFFORT=high`, `OLLAMA_TIMEOUT_SEC=420`.

```json
{
  "conversation_id": "...",
  "system_prompt": "Ты — внутренний IT-консультант...\n\n<documentation_context>\nДокумент: guide.md\n...",
  "messages": [
    {"role": "user", "body": "test"},
    {"role": "assistant", "body": "ответ"},
    {"role": "user", "body": "уточнение"}
  ]
}
```

В Ollama уходит `POST /api/chat` с `stream=false`, `messages`:
`system` + история `user`/`assistant` (`body` → `content`), `think`, `keep_alive=-1`
и `options` (`temperature`, `top_p`, `seed`, `num_predict`, `num_ctx`, `stop`).
Content — HTML-ответ, без JSON Schema и без проверки дословных цитат.
Дополнительные инструкции можно задать через `AI_SYSTEM_PROMPT` (контракт формата
ответа всё равно добавляется). Модель с Qdrant напрямую не общается — retrieval
делает `mail_gateway`.

При `LOG_LEVEL=INFO` reranker пишет начало и итог обработки: модель, число
кандидатов, время и лучший score. При `LOG_LEVEL=DEBUG` дополнительно видны
`source`, `chunk_index`, исходный vector score, итоговый rerank score и позиция
каждого чанка, а также время и token counters каждого запроса к Ollama. Текст
чанка в эти строки не выводится.

---

# Reindex

Сервис следит за каталогом документации (`WATCH_PATH`) и после паузы без новых
событий индексирует файлы в Qdrant: Docling convert → HybridChunker → Ollama
embeddings → upsert.

Поддерживаемые типы: `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.csv`, `.pdf`,
`.docx`, `.pptx`, `.xlsx`, `.xls`, `.html`, `.htm` (`pip install -e ".[reindex]"`).
Ридер возвращает семантические чанки Docling (`HybridChunker` + `contextualize`),
с заголовками секций. Каждая таблица Docling (`TableItem`) сериализуется целиком
в Markdown и кладётся одним Qdrant point без лимита `CHUNK_SIZE`. Для embedding
большая таблица локально делится на группы строк с общей шапкой и headings,
векторы групп нормализуются и усредняются; полный Markdown в payload не режется.
Повторные фрагменты HybridChunker и чанки только из `|---|` отбрасываются.
OCR для PDF выключен.

Картинки не гоняются через VLM-pipeline на всю страницу. Это **enrichment**:
обычный convert, затем Ollama VLM (`VLM_MODEL` на том же `OLLAMA_BASE_URL`)
описывает вырезанные рисунки. В чанк вместо `<!-- image -->` попадает блок
`[Изображение]` с описанием и подписью. HybridChunker и RAG не меняются.

## Структура

```
reindex/
  domain/             # DocumentChunk, суффиксы, обход файлов
  ports/              # DocumentReader, Indexer, Embedder
  adapters/           # Docling HybridChunker, QdrantIndexer, LoggingIndexer
  watcher.py          # watchdog + DebouncedReindex
  service.py          # composition root и run loop
mail_gateway/adapters/rag/
  qdrant_retriever.py
  ollama_reranker.py
  reranking_retriever.py
common/
  embeddings.py       # OllamaEmbedder (/api/embeddings)
deploy/               # install.sh + systemd units
tests/reindex/
```

## Поток

1. При старте (и при `--once`) — **сверка** каталога с Qdrant по SHA-256 содержимого
   и версии алгоритма индексации (коллекция не пересоздаётся). Уже
   проиндексированные актуальной версией пропускаются; файлы, которых нет на
   диске, снимаются из Qdrant.
2. Без `--once`: `watchdog` рекурсивно наблюдает `WATCH_PATH`.
3. События create / modify / delete / move копятся с debounce
   (`opened`/`closed` от чтения файлов игнорируются). Пока идёт проход,
   новые события ждут и применяются одним батчем — параллельных проходов нет.
4. После `DEBOUNCE_SECONDS` тишины индексируется **только затронутый файл**
   (create/modify), если его содержимое изменилось; байт-в-байт копии в других
   папках не эмбеддятся повторно (в индексе остаётся первый путь по сортировке).
   Delete снимает точки этого `source_path`; если удалён канонический файл,
   следующая копия с тем же хешем поднимается в индекс.
5. Ошибка в indexer логируется; сервис продолжает работать (`--once` пробрасывает ошибку).

## Конфиг (через общий `.env`)

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `WATCH_PATH` | Каталог документов (должен существовать) | `/var/lib/it-consultant/db` |
| `DEBOUNCE_SECONDS` | Пауза перед реиндексацией | `1.0` |
| `QDRANT_URL` | HTTP API Qdrant | `http://127.0.0.1:6333` |
| `QDRANT_COLLECTION` | Имя коллекции | `docs` |
| `EMBEDDING_MODEL` | Модель embeddings в Ollama | `nomic-embed-text` |
| `RAG_CANDIDATES` | Сколько кандидатов брать из Qdrant | `20` |
| `RAG_TOP_K` | Сколько чанков оставить после rerank | `8` |
| `RAG_NEIGHBOR_WINDOW` | Соседи в том же heading-разделе (±N); без headings — ±N по `chunk_index` | `1` |
| `RERANK_ENABLED` / `RERANK_MODEL` | Rerank через Ollama `POST /api/chat` (score `0..1`) | `true` / `dengcao/Qwen3-Reranker-8B:Q8_0` |
| `RERANK_NUM_PREDICT` | Лимит генерации reranker с учётом thinking; при исчерпании выполняется короткий retry без thinking | `256` |
| `CHUNK_SIZE` | Max tokens для прозы и одной embedding-части таблицы; полный table payload без лимита | `1024` |
| `PICTURE_DESCRIPTION_ENABLED` | VLM-описания картинок (enrichment) | `true` |
| `VLM_MODEL` | Vision-модель в том же Ollama | `qwen3-vl:8b` |
| `VLM_TIMEOUT_SEC` | Таймаут описания одной картинки | `90` |
| `VLM_CONCURRENCY` | Сколько картинок описывать параллельно | `2` |
| `PICTURE_AREA_THRESHOLD` | Мин. доля площади страницы для VLM | `0.02` |
| `LOG_LEVEL` | `INFO` — этапы файла; `DEBUG` — каждая картинка и HTTP библиотек | `INFO` |

`CHUNK_OVERLAP` в `.env` игнорируется reindex (соседей склеивает `merge_peers`).
Плюс общие / mail-поля из [`.env.example`](.env.example). `Settings` всё равно
требует `EWS_*` — для локального reindex достаточно заглушек.

## Запуск

```bash
docker compose up -d ollama qdrant
ollama pull nomic-embed-text
ollama pull qwen3-vl:8b   # или moondream / qwen2.5vl:3b для локальной проверки
pip install -e ".[reindex,dev]"
python -m reindex --once    # один проход, без watcher
python -m reindex           # watcher
```

## Проверка на Windows

Python — в WSL, Qdrant и Ollama — Docker Desktop (`localhost:6333` / `11434`
доступны из WSL).

1. `docker compose up -d qdrant ollama`
2. `ollama pull nomic-embed-text` и `ollama pull qwen3-vl:8b` (локально можно `VLM_MODEL=moondream`)
3. В `.env`: заглушки `EWS_*`, `WATCH_PATH` на существующую папку
   (из WSL: `/mnt/c/Users/.../docs`), `QDRANT_URL=http://127.0.0.1:6333`,
   `OLLAMA_BASE_URL=http://127.0.0.1:11434`
4. В WSL: `pip install -e ".[reindex]"` (если No space — сжать VHDX WSL;
   конвертер идёт без OCR)
5. Положить в `WATCH_PATH` смесь `.md`, `.txt`, `.pdf`, `.docx`, `.xlsx`
6. `python -m reindex --once` — в логе `Qdrant reindex done ... points=N`
7. Дашборд: http://127.0.0.1:6333/dashboard — коллекция `docs`, payload `text`
   с заголовками секций, таблицы в Markdown, картинки как `[Изображение]`
8. Для проверки watcher: `python -m reindex`, затем добавить/заменить/удалить
   файл — в логе `apply_changes` только для этого пути, коллекция не
   пересоздаётся

## Indexer

По умолчанию используется `QdrantIndexer`. Stub `LoggingIndexer` остаётся для тестов.
На старте / `--once` — reconcile по SHA-256 содержимого и версии индексатора
(skip неизменённых, dedup копий 1:1).
Watcher вызывает `apply_changes`: upsert одного изменившегося файла или delete по
`source_path`. Payload точки: `text`, `source_path`, `chunk_index` (совместимо с RAG),
плюс `headings`, `file_hash`, `index_version`, `chunk_type`; для таблиц также
`table_ref` и `row_count`.

## Тесты

```bash
pytest tests/reindex tests/common tests/deploy
```

Интеграционные тесты поднимают настоящий watcher на временном каталоге и проверяют create/modify/delete/move по конкретным путям, debounce-батч и устойчивость к исключениям indexer. Тесты деплоя ставят сервисы в фейковый root (`--dest-dir`), без `systemctl` и без записи в настоящий `/etc`.

## systemd

Unit-файлы: [`deploy/systemd/`](deploy/systemd/). Установка и снятие — через [`deploy/install.sh`](deploy/install.sh) (`--enable` / `--only reindex` / `--undeploy`, см. раздел «Установка на сервер» выше).
