FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DRIFT_DATABASE_PATH=/data/drift.db

WORKDIR /app
COPY pyproject.toml ./
COPY drift.py ./
COPY monitoring ./monitoring
RUN pip install --no-cache-dir .

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "monitoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
