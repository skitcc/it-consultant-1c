from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    conversation_id: str
    item_id: str
    change_key: str
    from_address: str
    subject: str
    body: str
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class Reply:
    conversation_id: str
    in_reply_to_item_id: str
    in_reply_to_change_key: str
    body: str
