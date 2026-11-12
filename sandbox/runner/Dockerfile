# The sandbox runner: a small web service that starts one locked down container per run through the
# Docker client. It is the only container that holds the Docker socket and it holds no other secret.
FROM docker:29-cli@sha256:2357639ad1ea7d0194134bf3292622fd4e562483966ed674e5d7c945b5f3b28d AS docker-client

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=docker-client /usr/local/bin/docker /usr/local/bin/docker
COPY requirements.lock runner.py /app/
WORKDIR /app
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

HEALTHCHECK --interval=5s --timeout=5s --start-period=2s --start-interval=1s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8200/healthz')"

CMD ["uvicorn", "runner:create_app", "--factory", "--host", "0.0.0.0", "--port", "8200"]
