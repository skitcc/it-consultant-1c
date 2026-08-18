from __future__ import annotations

import uvicorn

from api_gateway.app import create_app
from api_gateway.settings import GatewaySettings


def main() -> None:
    settings = GatewaySettings()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.api_gateway_host,
        port=settings.api_gateway_port,
        workers=1,
    )


if __name__ == "__main__":
    main()
