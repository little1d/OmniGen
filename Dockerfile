FROM node:24-alpine AS frontend

WORKDIR /build/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    OMNIGEN_DATA_PATH=/data/items.json \
    OMNIGEN_REVIEW_DB=/data/reviews/reviews.sqlite3

WORKDIR /app
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --retries 5 fastapi==0.141.1 uvicorn==0.44.0 \
    && mkdir -p /data/reviews

COPY src/ ./src/
COPY dashboard/sample-data/ ./dashboard/sample-data/
COPY --from=frontend /build/dashboard/dist/ ./dashboard/dist/

ENV OMNIGEN_DATA_PATH=/app/dashboard/sample-data/items.json

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "src.review_app:app", "--host", "0.0.0.0", "--port", "8000"]
