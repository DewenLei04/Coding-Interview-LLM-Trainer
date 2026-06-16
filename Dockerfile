FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend
COPY data ./data

ARG INSTALL_EXTRAS=dev
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[${INSTALL_EXTRAS}]"

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "coding_interview_trainer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
