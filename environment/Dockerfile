FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pytest tmux asciinema \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /app /logs/verifier \
    && chown -R worker:worker /app /logs

WORKDIR /app
COPY app/ /app/
RUN chmod 0755 /app/regression_recover.py

USER worker
CMD ["sleep", "infinity"]
