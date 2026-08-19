from fastapi.testclient import TestClient

from api_gateway.app import create_app
from api_gateway.settings import GatewaySettings
from knowledge.bootstrap import KnowledgeContainer
from knowledge.settings import KnowledgeSettings


class DummyAnswer:
    def execute(self, question, **kwargs):
        return "ok"


class DummyRegistry:
    def list(self, knowledge_id: str = "main"):
        return []


def test_gateway_settings_do_not_require_owui_loader_key() -> None:
    settings = GatewaySettings(
        _env_file=None,
        API_GATEWAY_API_KEY="chat-secret",
    )
    assert settings.api_gateway_api_key == "chat-secret"


def test_create_app_does_not_expose_process_loader() -> None:
    settings = GatewaySettings(_env_file=None, API_GATEWAY_API_KEY="chat-secret")
    container = KnowledgeContainer(
        settings=KnowledgeSettings(_env_file=None),
        registry=DummyRegistry(),
        vector_index=object(),
        index_document=object(),
        remove_document=object(),
        update_metadata=object(),
        retrieve_knowledge=object(),
        answer_question=DummyAnswer(),
    )
    app = create_app(settings=settings, container=container)
    client = TestClient(app)
    assert client.put("/process").status_code == 404
    models = client.get("/v1/models", headers={"Authorization": "Bearer chat-secret"})
    assert models.status_code == 200
    assert client.get("/health").status_code == 200
