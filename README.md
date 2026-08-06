# IT Consultant

Репозиторий IT-консультанта: почтовый шлюз и сервис реиндексации файловой БД.

## Компоненты

| Пакет | Назначение |
|-------|------------|
| `common` | Общие настройки (`Settings`) и утилиты |
| `mail_gateway` | Exchange (EWS Streaming) → Qdrant RAG → Ollama → reply |
| `reindex` | Следит за каталогом документов и индексирует их в Qdrant |

Общий конфиг — один класс [`common.Settings`](common/settings.py), читается из `.env` / переменных окружения. Оба сервиса используют одни и те же переменные.

## Установка (разработка)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
# для reindex также:
pip install -e ".[reindex,dev]"
cp .env.example .env
```

Заполните в `.env` поля `EWS_*` (и при необходимости `WATCH_PATH` / `DEBOUNCE_SECONDS`).

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
sudo ./deploy/install.sh --enable

# только разложить файлы, без enable:
sudo ./deploy/install.sh
sudo systemctl daemon-reload
sudo systemctl enable --now it-consultant.target
```

`enable it-consultant.target` создаёт symlink’и в `multi-user.target.wants` и (через `Also=`) подтягивает `mail-gateway` и `reindex`.

### Безопасная проверка без трогания host rootfs

Никогда не гоняйте «тестовый» install в `/` без `--dest-dir`. Используйте фейковый root:

```bash
./deploy/install.sh --dest-dir /tmp/itc-root --layout-only
# дерево только под /tmp/itc-root/... ; systemctl и useradd не вызываются

./deploy/install.sh --dest-dir /tmp/itc-root   # + venv/pip внутрь /tmp/...
```

Автотесты (тоже через `--dest-dir` во временный каталог):

```bash
pytest tests/deploy -q          # layout, содержимое unit-файлов
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
4. Embedding вопроса → Qdrant `RAG_CANDIDATES` → rerank → `RAG_TOP_K` (+ соседние чанки) → фрагменты в `system_prompt`.
5. `POST` в Ollama `/api/chat`.
6. Reply через EWS в тот же тред.
7. При обрыве streaming — reconnect.

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
3. Проиндексируйте документы (`WATCH_PATH`) через `python -m reindex`.
4. Запуск шлюза:
   ```bash
   python -m mail_gateway
   ```
5. Напишите письмо на ящик бота. В логах будет `Assistant payload` (с `system_prompt` и фрагментами документации) и ответ модели.

## Контракт Ollama

Шлюз подтягивает тред из Exchange, чистит тела, достаёт кандидатов из Qdrant,
переранжирует их (если доступен `/api/rerank` или `/v1/rerank`), дополняет соседними
чанками и логирует внутренний payload:

```json
{
  "conversation_id": "...",
  "system_prompt": "Ты IT-консультант...\n\nРелевантные фрагменты документации:\n[1] source=guide.md\n...",
  "messages": [
    {"role": "user", "body": "test"},
    {"role": "assistant", "body": "ответ"},
    {"role": "user", "body": "уточнение"}
  ]
}
```

В Ollama уходит `POST /api/chat` с `messages`: `system` + история `user`/`assistant`
(`body` → `content`). Системный промпт можно переопределить через `AI_SYSTEM_PROMPT`.
Модель с Qdrant напрямую не общается — retrieval делает `mail_gateway`.

---

# Reindex

Сервис следит за каталогом документации (`WATCH_PATH`) и после паузы без новых
событий индексирует файлы в Qdrant: parse → chunk → Ollama embeddings → upsert.

Поддерживаемые типы: `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.csv`, а также
`.pdf` / `.docx` (нужен `pip install -e ".[reindex]"`).

## Структура

```
reindex/
  ports.py            # DocumentReader
  adapters/           # Text/Pdf/Docx + CompositeDocumentReader
  documents.py        # обход файлов по суффиксу
  indexer.py          # ABC Indexer + LoggingIndexer (stub для тестов)
  qdrant_indexer.py   # QdrantIndexer
  watcher.py          # watchdog + DebouncedReindex
  service.py          # composition root и run loop
mail_gateway/adapters/rag/
  qdrant_retriever.py
  ollama_reranker.py
  reranking_retriever.py
common/
  embeddings.py       # OllamaEmbedder (/api/embeddings)
  chunking.py         # разбиение текста на чанки
deploy/               # install.sh + systemd units
tests/reindex/
```

## Поток

1. При старте — полный reindex каталога.
2. `watchdog` рекурсивно наблюдает `WATCH_PATH`.
3. События create / modify / delete / move сбрасывают debounce-таймер.
4. После `DEBOUNCE_SECONDS` тишины — снова полный reindex в коллекцию Qdrant.
5. Ошибка в indexer логируется; сервис продолжает работать.

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
| `RAG_NEIGHBOR_WINDOW` | Соседние chunk_index (±N) | `1` |
| `RERANK_ENABLED` / `RERANK_MODEL` | Rerank через Ollama-compatible API | `true` / `bge-reranker-v2-m3` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Размер чанка и overlap (символы) | `1200` / `150` |
| `LOG_LEVEL` | Уровень логов | `INFO` |

Плюс общие / mail-поля из [`.env.example`](.env.example).

## Запуск

```bash
docker compose up -d ollama qdrant
ollama pull nomic-embed-text
pip install -e ".[reindex,dev]"
python -m reindex
```

## Indexer

По умолчанию используется `QdrantIndexer`. Stub `LoggingIndexer` остаётся для тестов.

## Тесты

```bash
pytest tests/reindex tests/common tests/deploy
```

Интеграционные тесты поднимают настоящий watcher на временном каталоге и проверяют create/modify/delete/move, вложенные пути, debounce и устойчивость к исключениям indexer. Тесты деплоя ставят сервисы в фейковый root (`--dest-dir`), без `systemctl` и без записи в настоящий `/etc`.

## systemd

Unit-файлы: [`deploy/systemd/`](deploy/systemd/). Установка — через [`deploy/install.sh`](deploy/install.sh) (см. раздел «Установка на сервер» выше).
