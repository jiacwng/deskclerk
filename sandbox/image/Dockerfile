# The image every sandbox run starts from: Python with the data libraries and nothing else. It holds
# no secret and the runner starts it with no network, a read-only filesystem and no capabilities.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.lock entrypoint.py /opt/sandbox/
RUN pip install --no-cache-dir --require-hashes -r /opt/sandbox/requirements.lock

WORKDIR /mnt/data
ENTRYPOINT ["python", "/opt/sandbox/entrypoint.py"]
