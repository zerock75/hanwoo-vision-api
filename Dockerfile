FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
		  language-pack-ko \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

# Install deps against an empty package so this ~6GB layer is cached across
# source edits; the editable install resolves to /app/src, filled in below.
RUN mkdir -p src/hanwoo \
    && python3 -m pip install --no-cache-dir --break-system-packages -e .

COPY src ./src

RUN python3 -m pip install --no-cache-dir --break-system-packages --no-deps -e .



EXPOSE 8000 8001
