# The deskclerk web app, installed from hash-locked requirements. Settings come from DESKCLERK_*
# variables, the server list is mounted at /app/servers.toml and the token store lives in the
# /app/data volume.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock requirements-build.lock pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --require-hashes -r requirements-build.lock -r requirements.lock \
    && pip install --no-cache-dir --no-deps --no-build-isolation . \
    && useradd --create-home --uid 10001 deskclerk \
    && mkdir data \
    && chown deskclerk:deskclerk data

USER deskclerk

HEALTHCHECK --interval=2s --timeout=5s --retries=30 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('DESKCLERK_PORT', '8000'))"

CMD ["python", "-m", "deskclerk"]
