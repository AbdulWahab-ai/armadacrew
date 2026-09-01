FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARMADACREW_HOST=0.0.0.0 \
    ARMADACREW_PORT=8080 \
    ARMADACREW_DB_PATH=/var/lib/armadacrew/armada.db \
    ARMADACREW_DATA_DIR=/app/data \
    ARMADACREW_LLM_PROVIDER=mock

COPY pyproject.toml README.md LICENSE ./
COPY armadacrew ./armadacrew
COPY static ./static
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && mkdir -p /var/lib/armadacrew

EXPOSE 8080

CMD ["uvicorn", "armadacrew.api:app", "--host", "0.0.0.0", "--port", "8080"]
