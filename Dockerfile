FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md drift.py ./
COPY monitoring ./monitoring
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DRIFT_DATABASE_PATH=/data/drift.db

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin monitor \
    && mkdir /data \
    && chown monitor:monitor /data
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

USER monitor
WORKDIR /home/monitor
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "monitoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
