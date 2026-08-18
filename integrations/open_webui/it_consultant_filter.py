"""
title: IT Consultant external RAG
description: Prevent Open WebUI from injecting its own file context.
version: 1.0.0
"""

from pydantic import BaseModel

# Open WebUI reads this value from the module, not the Filter instance.
file_handler = True


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def inlet(self, body: dict) -> dict:
        # Retrieval and prompt construction happen inside API Gateway.
        return body
