# IT Consultant

Универсальный RAG-сервис для Open WebUI и Exchange:

```text
Open WebUI (upload) ─> bind-mount uploads/ ─> reindex (watchdog)
                                              └─> Knowledge Core ─> Qdrant / Ollama

Open WebUI / Exchange ─> API Gateway ─> Knowledge Core ─> Qdrant / Ollama
```

Open WebUI только сохраняет файлы на диск и отправляет чат в gateway.
Индексацию делает хостовый `reindex`: inotify/watchdog, затем `IndexDocument`.

## Компоненты

- `knowledge/core` — независимые domain models, ports и use cases.
- `knowledge/adapters` — Docling, Ollama, Qdrant и SQLite.
- `api_gateway` — OpenAI-compatible RAG chat (`it-consultant`).
- `reindex` — следит за `WATCH_PATH` и индексирует create/update/delete.
- `mail_gateway` — EWS transport, вызывающий API Gateway.

Open WebUI не эмбеддит документы сам (`BYPASS_EMBEDDING_AND_RETRIEVAL`).
External Document Loader и HTTP `knowledge_sync` не используются.
Обычное изменение никогда не пересоздаёт Qdrant collection и затрагивает
только points конкретного `document_id`.

Имена файлов OWUI в `uploads/` выглядят как `{file_uuid}_{оригинал.pdf}`.
Watcher берёт uuid как `document_id`, а в цитатах оставляет оригинальное имя.

## Локальная установка

Все Python-команды выполняются через `.venv`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
.venv/bin/python -m pip install -e ".[api,reindex,dev]"
cp .env.example .env
```

Docling использует CPU PyTorch. CUDA wheels не нужны на CPU-сервере.
Эмбеддинги и чат идут через Ollama (GPU в контейнере Ollama).

Создайте каталоги, которые видят и Docker, и systemd:

```bash
sudo mkdir -p /var/lib/it-consultant/owui-data/uploads
sudo chown -R "$USER:$USER" /var/lib/it-consultant
```

## Настройка Open WebUI

В Admin → Documents:

```text
Bypass Embedding and Retrieval: enabled
Hybrid Search: disabled
```

Не включайте Content Extraction Engine = external: файлы должны только
попадать в `/app/backend/data/uploads` (хост: `WATCH_PATH`).

В Admin → Connections → OpenAI:

```text
Base URL: http://api-gateway:8000/v1
API Key: значение API_GATEWAY_API_KEY
```

Gateway показывает одну виртуальную модель `it-consultant`. Она представляет
полный pipeline retrieval → rerank → prompt → Ollama. Прямые модели Ollama
остаются отдельными моделями без этого RAG.

Filter Function из `integrations/open_webui/it_consultant_filter.py` необходимо
прикрепить только к модели `it-consultant`. Его `file_handler = True` запрещает
Open WebUI добавлять собственный RAG-контекст.

## API Gateway

Минимальный внешний API:

- `GET /v1/models` — модель `it-consultant`.
- `POST /v1/chat/completions` — OpenAI-compatible RAG chat.
- `GET /health` — liveness.
- `GET /ready` — readiness инфраструктуры.

Отдельных document-management и retrieve endpoints нет.

## Reindex

`reindex` при старте сверяет диск с registry/Qdrant, затем смотрит `WATCH_PATH`
через watchdog. События `opened`/`closed` игнорируются, чтобы чтение файла
не зациклило индексацию. После паузы `DEBOUNCE_SECONDS` применяется пачка
upsert/delete.

Open WebUI v0.11 при удалении из Knowledge часто не стирает блоб в `uploads/`.
Поэтому watcher также следит за `webui.db` и `webui.db-wal` (каталог DATA_DIR
рядом с `uploads/`). SQLite в WAL-режиме пишет удаления в `-wal`, а основной
файл может не меняться часами. `-shm` не смотрим: его трогают и наши чтения.
Пустой WAL-heartbeat пропускается, если id/`hash`/`updated_at` в таблице
`file` не изменились.
Правка документа в OWUI не трогает блоб в `uploads/` — текст лежит в
`file.data.content`. Watcher подхватывает смену `hash`/`updated_at` и
индексирует этот текст.
Одинаковое содержимое индексируется один раз (канонический путь — первый
по имени); удаление канонической копии поднимает следующую.
После изменения SQLite сироты с именем `{uuid}_...`, которых уже нет в таблице
`file`, удаляются с диска — обычный inotify `deleted` убирает их из Qdrant.
Ручные копии без uuid-префикса не трогаются. Периодический скан не используется.

```bash
.venv/bin/python -m reindex --once
.venv/bin/python -m reindex
```

Compose **не** запускает indexer: один процесс на хосте, один inotify.

## Mail Gateway

Почтовый сервис больше не подключается к Qdrant/Ollama напрямую:

```text
EWS message
→ clean/load thread
→ POST API_GATEWAY_BASE_URL/chat/completions
→ reply into the same EWS conversation
```

Обязательные параметры:

```dotenv
API_GATEWAY_BASE_URL=http://127.0.0.1:8000/v1
API_GATEWAY_API_KEY=change-chat-key
API_GATEWAY_MODEL=it-consultant
```

## Docker Compose

```bash
docker compose up -d --build ollama qdrant api-gateway open-webui
```

Compose поднимает Ollama, Qdrant, Open WebUI и API Gateway.
Open WebUI зафиксирован на конкретной версии вместо `:main`.
Данные OWUI монтируются с хоста (`OWUI_DATA_DIR`, по умолчанию
`/var/lib/it-consultant/owui-data`), registry — из `ITC_VAR_DIR`
(`/var/lib/it-consultant`), чтобы хостовый `reindex` видел те же файлы
и SQLite.

Перед запуском:

1. заполните `.env` (`API_GATEWAY_API_KEY` обязателен для Compose);
2. создайте `WATCH_PATH` на хосте;
3. загрузите модели в Ollama (`nomic-embed-text`, чат-модель);
4. запустите `python -m reindex` или systemd-unit.

## Systemd deployment

```bash
sudo ./deploy/install.sh --enable

# Или запустить только один процесс:
sudo ./deploy/install.sh --only api-gateway --enable
sudo ./deploy/install.sh --only reindex --enable
sudo ./deploy/install.sh --only mail-gateway --enable
```

Устанавливаются:

```text
/opt/it-consultant/.venv
/etc/it-consultant/.env
/var/lib/it-consultant/registry.sqlite3
/var/lib/it-consultant/owui-data/uploads
/etc/systemd/system/api-gateway.service
/etc/systemd/system/reindex.service
/etc/systemd/system/mail-gateway.service
/etc/systemd/system/it-consultant.target
```

Fake-root проверка:

```bash
./deploy/install.sh --dest-dir /tmp/itc-root --layout-only
./deploy/install.sh --dest-dir /tmp/itc-root --undeploy
```

## Тесты

```bash
.venv/bin/python -m pytest -q
```

Ключевые проверки:

- binary body не изменяется между indexer и Docling;
- update одного документа не затрагивает остальные points;
- ошибка новой версии сохраняет предыдущую;
- watcher не реагирует на opened/closed и debounce-ит пачку файлов;
- upload `{uuid}_{name}` даёт тот же `document_id`, что uuid OWUI;
- удаление в OWUI Knowledge чистит сирот в `uploads/` через watch `webui.db`;
- OWUI и mail используют один `AnswerQuestion`;
- `knowledge/core` не импортирует инфраструктурные библиотеки.
