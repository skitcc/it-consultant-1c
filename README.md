# Mail Gateway

Почтовый шлюз IT-консультанта: Exchange (EWS Streaming) → ИИ-сервис → reply в тот же conversation.

## Структура

```
mail_gateway/
  domain/       # модели (IncomingMessage, Reply)
  ports/        # контракты MailListener, MailSender, Assistant
  application/  # сценарий HandleIncomingMail
  adapters/     # реализации портов (EWS, HTTP/Stub ИИ)
  main/         # конфиг и composition root
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

## Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Заполните в `.env` поля `EWS_*` (адреса через туннель с VPS).

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
