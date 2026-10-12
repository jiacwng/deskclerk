# The image behind the API, the MCP server and the one-shot data load.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEEDTOKNOW_MODEL_CACHE=/models

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# The embedding model is downloaded here, so a started container needs no network.
RUN pip install --no-cache-dir . \
    && python -c "from needtoknow.embed import embed_query; embed_query('warm up')" \
    && useradd --create-home --uid 10001 needtoknow \
    && chown -R needtoknow:needtoknow /models

USER needtoknow

EXPOSE 8000 8001

CMD ["uvicorn", "--factory", "needtoknow.api:create_default_app", \
     "--host", "0.0.0.0", "--port", "8000"]
