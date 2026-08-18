FROM python:3.12-slim

ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/it-consultant

COPY pyproject.toml README.md LICENSE ./
COPY common common
COPY knowledge knowledge
COPY api_gateway api_gateway
COPY knowledge_sync knowledge_sync
COPY mail_gateway mail_gateway
COPY integrations integrations

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_CPU_INDEX}" torch torchvision \
    && python -m pip install ".[api]"

RUN useradd --system --uid 10001 --create-home it-consultant \
    && mkdir -p /var/lib/it-consultant \
    && chown -R it-consultant:it-consultant /opt/it-consultant /var/lib/it-consultant

USER it-consultant

CMD ["python", "-m", "api_gateway"]
