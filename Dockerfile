FROM python:3.12-slim
LABEL authors="amine"

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV OSUNATOR_ROOT=/app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


COPY src/ src/
COPY best_model.keras ./
COPY data_meta/norm_stats.json data_meta/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "osunator.cli"]