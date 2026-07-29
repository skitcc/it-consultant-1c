# IT Consultant

Репозиторий IT-консультанта: почтовый шлюз и сервис реиндексации файловой БД.

## Компоненты

| Пакет | Назначение |
|-------|------------|
| `common` | Общие настройки (`Settings`) и утилиты |
| `mail_gateway` | Exchange (EWS Streaming) → ИИ-сервис → reply в тот же conversation |
| `reindex` | Следит за каталогом файловой БД и при изменениях запускает реиндексацию |

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

Почтовый шлюз: Exchange (EWS Streaming) → ИИ-сервис → reply в тот же conversation.

## Структура

```
common/         # Settings, logging
mail_gateway/
  domain/       # модели (IncomingMessage, Reply)
  ports/        # контракты MailListener, MailSender, Assistant
  application/  # сценарий HandleIncomingMail
  adapters/     # реализации портов (EWS, HTTP/Stub ИИ)
  main/         # composition root
tests/
```

Зависимости направлены внутрь: `adapters` → `ports`/`domain`, `application` → `ports`/`domain`.  
Адаптеры явно наследуют порты (`class EwsMailListener(MailListener)`).

## Поток

1. EWS Streaming: событие `NewMail` в Inbox.
2. Чтение письма → `conversation_id`, `item_id`, `change_key`, текст.
3. HTTP POST в ИИ (или stub).
4. Reply через EWS в тот же тред.
5. При обрыве streaming — reconnect.

`change_key` — версия объекта письма в Exchange. Вместе с `item_id` однозначно указывает на конкретную ревизию письма; без него `get`/`reply` могут упасть, если письмо уже изменилось.

## Как протестировать самому

### 1. Unit-тесты (без Exchange)

```bash
pytest
```

Проверяют сценарий «письмо → ИИ → reply» на фейковых портах.

### 2. Round-trip с реальным ящиком (stub ИИ)

1. В `.env`:
   - `ASSISTANT_MODE=stub`
   - рабочие `EWS_SERVER`, `EWS_EMAIL`, `EWS_PASSWORD`
2. Запуск:
   ```bash
   python -m mail_gateway
   ```
3. Напишите письмо на ящик бота (или reply в существующий тред).
4. В логах должны появиться `New mail ...` и `Reply sent ...`.
5. В Outlook должен прийти stub-ответ с `conversation_id`.

### 3. С HTTP-заглушкой ИИ

Поднимите любой mock, который отвечает `{ "reply": "тест" }` на `POST /v1/ask`, затем:

```env
ASSISTANT_MODE=http
AI_SERVICE_URL=http://127.0.0.1:8000/v1/ask
```

## Контракт ИИ

`POST` на `AI_SERVICE_URL`:

```json
{
  "conversation_id": "...",
  "from": "user@company.ru",
  "subject": "...",
  "body": "..."
}
```

Ответ: `{ "reply": "текст" }` или `{ "reply": null }` → текст про администратора.

---

# Reindex

Сервис следит за каталогом файловой базы данных (рекурсивно: файлы и поддиректории) и после паузы без новых событий вызывает реиндексацию.

Настоящая индексация пока не реализована: есть абстрактный `Indexer` и stub `LoggingIndexer`, который пишет debug в лог.

## Структура

```
reindex/
  indexer.py     # ABC Indexer + LoggingIndexer (stub)
  watcher.py     # watchdog + DebouncedReindex
  service.py     # composition root и run loop
deploy/          # install.sh + systemd units
tests/reindex/   # интеграционные и unit-тесты
```

## Поток

1. `watchdog` рекурсивно наблюдает `WATCH_PATH`.
2. События create / modify / delete / move (файлы и директории) сбрасывают debounce-таймер.
3. После `DEBOUNCE_SECONDS` тишины вызывается `Indexer.reindex(watch_path)`.
4. Ошибка в indexer логируется; сервис продолжает работать.

## Конфиг (через общий `.env`)

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `WATCH_PATH` | Каталог файловой БД (должен существовать) | `/var/lib/it-consultant/db` |
| `DEBOUNCE_SECONDS` | Пауза перед реиндексацией после последнего события | `1.0` |
| `LOG_LEVEL` | Уровень логов | `INFO` |

Плюс общие / mail-поля из [`.env.example`](.env.example) (один файл на оба сервиса).

## Запуск

```bash
pip install -e ".[reindex,dev]"
python -m reindex
```

## Indexer

```python
class Indexer(ABC):
    def reindex(self, watch_path: str) -> None: ...

class LoggingIndexer(Indexer):
    # stub: только debug-лог, без реальной индексации
```

В `main` по умолчанию используется `LoggingIndexer`. Позже сюда подставится реальная реализация.

## Тесты

```bash
pytest tests/reindex tests/common tests/deploy
```

Интеграционные тесты поднимают настоящий watcher на временном каталоге и проверяют create/modify/delete/move, вложенные пути, debounce и устойчивость к исключениям indexer. Тесты деплоя ставят сервисы в фейковый root (`--dest-dir`), без `systemctl` и без записи в настоящий `/etc`.

## systemd

Unit-файлы: [`deploy/systemd/`](deploy/systemd/). Установка — через [`deploy/install.sh`](deploy/install.sh) (см. раздел «Установка на сервер» выше).
