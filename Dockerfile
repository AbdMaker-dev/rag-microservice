# Multi-stage so the runtime image carries no build toolchain.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# La voix des cours (fr_FR-siwis-medium, ~60 Mo) se télécharge au build :
# en production, le conteneur n'a pas d'accès internet sortant garanti.
RUN /opt/venv/bin/python -m piper.download_voices fr_FR-siwis-medium \
      --data-dir /build/voices


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ffmpeg compresse la voix en MP3 (mono 64 kbit/s) ; sans lui, /speech
# répondrait en WAV, dix fois plus lourd.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Never run as root: the monorepo applies the same rule to its images.
RUN useradd --create-home --uid 10001 rag
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=rag:rag /build/voices ./voices
COPY --chown=rag:rag app ./app
COPY --chown=rag:rag migrations ./migrations

USER rag
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
