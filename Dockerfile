# The deskclerk web app. Settings come from DESKCLERK_* variables, the server list is mounted at
# /app/servers.toml and the token store lives in the /app/data volume.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir '.[tracing]' \
    && useradd --create-home --uid 10001 deskclerk \
    && mkdir data \
    && chown deskclerk:deskclerk data

USER deskclerk

HEALTHCHECK --interval=2s --timeout=5s --retries=30 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('DESKCLERK_PORT', '8000'))"

CMD ["python", "-m", "deskclerk"]
