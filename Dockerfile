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

# RUN python3 -m pip install --no-cache-dir --break-system-packages -e .

RUN python3 - << 'EOF'
import tomllib, subprocess, sys
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
subprocess.run([sys.executable, '-m', 'pip', 'install',
    '--no-cache-dir', '--break-system-packages'] + deps, check=True)
EOF

COPY src ./src

RUN python3 -m pip install --no-cache-dir --break-system-packages --no-deps -e .



EXPOSE 8000 8001
