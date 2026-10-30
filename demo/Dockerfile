# The demo server: the official MCP git server (mcp-server-git) on a clone of the public deskclerk
# repository, served over Streamable HTTP on port 8100 by mcp-proxy. Writes stay in the clone.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_AUTHOR_NAME="deskclerk demo" \
    GIT_AUTHOR_EMAIL="demo@localhost" \
    GIT_COMMITTER_NAME="deskclerk demo" \
    GIT_COMMITTER_EMAIL="demo@localhost"

COPY requirements.lock /tmp/requirements.lock

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && useradd --create-home --uid 10001 demo \
    && mkdir /sandbox \
    && chown demo /sandbox

USER demo

RUN git clone --depth 100 https://github.com/jiacwng/deskclerk /sandbox/deskclerk

CMD ["mcp-proxy", "--host", "0.0.0.0", "--port", "8100", "--transport", "streamablehttp", "--", \
     "mcp-server-git", "--repository", "/sandbox/deskclerk"]
