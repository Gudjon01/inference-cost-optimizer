# Build from the repo root: docker build -f py/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /bin/uv
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /opt/ico/py
# Dependencies first, so editing source does not reinstall them.
COPY py/pyproject.toml py/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY py/src ./src
RUN uv sync --frozen --no-dev

COPY schema /opt/ico/schema
ENV PATH=/opt/ico/py/.venv/bin:$PATH ICO_SCHEMA_DIR=/opt/ico/schema
USER nobody
EXPOSE 4317
ENTRYPOINT ["ico"]
CMD ["collector"]
