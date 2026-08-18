# IT Consultant

Универсальный RAG-сервис для Open WebUI и Exchange:

```text
Open WebUI ─┐
            ├─> API Gateway ─> Knowledge Core ─> Qdrant / Ollama
Exchange ───┘

Open WebUI Knowledge ─> knowledge_sync ─> Knowledge Core
```

## Компоненты

- `knowledge/core` — независимые domain models, ports и use cases.
- `knowledge/adapters` — Docling, Ollama, Qdrant и SQLite.
- `api_gateway` — `/process` и OpenAI-compatible RAG chat.
- `knowledge_sync` — инкрементальная сверка одной OWUI Knowledge-базы.
- `mail_gateway` — EWS transport, вызывающий API Gateway.

Folder watcher и сервис `reindex` удалены. Новый документ индексируется синхронно
во время `PUT /process`; sync-сервис обрабатывает пропущенные upload, update,
rename и delete. Обычное изменение никогда не пересоздаёт Qdrant collection и
затрагивает только points конкретного `document_id`.

## Целостность upload

Open WebUI External Document Loader читает сохранённый файл в binary mode и
отправляет raw HTTP body. Gateway принимает body как `bytes`, а Docling adapter:

1. вычисляет SHA-256 полученных bytes;
2. записывает их без преобразований во временный файл с исходным suffix;
3. повторно вычисляет SHA-256 временного файла;
4. вызывает `DocumentConverter` только при совпадении hashes.

JSON/base64 и декодирование документа в этой цепочке не используются.

## Локальная установка

Все Python-команды выполняются через `.venv`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
.venv/bin/python -m pip install -e ".[api,dev]"
cp .env.example .env
```

Docling использует CPU PyTorch. CUDA wheels не нужны на CPU-сервере.

## Настройка Open WebUI

Создайте одну Knowledge-базу и сохраните её ID в:

```dotenv
OPEN_WEBUI_KNOWLEDGE_ID=<knowledge-id>
OPEN_WEBUI_SYNC_TOKEN=<service-account-api-token>
```

В Admin → Documents:

```text
Content Extraction Engine: external
External Loader URL: http://api-gateway:8000
External Loader API Key: значение OWUI_LOADER_KEY
Bypass Embedding and Retrieval: enabled
Hybrid Search: disabled
```

Custom headers:

```json
{
  "X-OpenWebUI-File-Id": "{{FILE_ID}}",
  "X-OpenWebUI-File-Name": "{{FILE_NAME}}"
}
```

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

- `PUT /process` — OWUI External Document Loader; успех только после Qdrant.
- `GET /v1/models` — модель `it-consultant`.
- `POST /v1/chat/completions` — OpenAI-compatible RAG chat.
- `GET /health` — liveness.
- `GET /ready` — readiness инфраструктуры.

Отдельных document-management и retrieve endpoints нет.

## Knowledge sync

`knowledge_sync` получает metadata всех файлов Knowledge, но скачивает и
индексирует только:

- новый file ID, отсутствующий в registry;
- файл с изменившимся hash;
- файл, callback которого был пропущен во время downtime.

Неизменённые файлы не скачиваются, не парсятся и не эмбеддятся. Rename обновляет
только Qdrant payload. Delete применяется после нескольких успешных snapshots;
при недоступном OWUI удаления запрещены.

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
docker compose up -d --build
```

Compose запускает Ollama, Qdrant, Open WebUI, API Gateway и knowledge sync.
Open WebUI зафиксирован на конкретной версии вместо `:main`.

Перед запуском:

1. заполните `.env`;
2. создайте пользователя/Knowledge в OWUI;
3. выпустите API token с доступом на чтение Knowledge;
4. установите `OPEN_WEBUI_KNOWLEDGE_ID` и `OPEN_WEBUI_SYNC_TOKEN`;
5. загрузите нужные модели в Ollama.

## Systemd deployment

```bash
sudo ./deploy/install.sh --enable

# Или запустить только один процесс:
sudo ./deploy/install.sh --only api-gateway --enable
sudo ./deploy/install.sh --only knowledge-sync --enable
sudo ./deploy/install.sh --only mail-gateway --enable
```

Устанавливаются:

```text
/opt/it-consultant/.venv
/etc/it-consultant/.env
/var/lib/it-consultant/registry.sqlite3
/etc/systemd/system/api-gateway.service
/etc/systemd/system/knowledge-sync.service
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

- binary body не изменяется между Gateway и Docling;
- update одного документа не затрагивает остальные points;
- ошибка новой версии сохраняет предыдущую;
- Knowledge sync не переиндексирует неизменённые документы;
- OWUI и mail используют один `AnswerQuestion`;
- `knowledge/core` не импортирует инфраструктурные библиотеки.
