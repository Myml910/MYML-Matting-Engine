FROM python:3.10-slim

ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://mirrors.nju.edu.cn/pytorch/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -ri "s@https?://deb.debian.org/debian@${DEBIAN_MIRROR}/debian@g; s@https?://security.debian.org/debian-security@${DEBIAN_MIRROR}/debian-security@g" /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -ri "s@https?://deb.debian.org/debian@${DEBIAN_MIRROR}/debian@g; s@https?://security.debian.org/debian-security@${DEBIAN_MIRROR}/debian-security@g" /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1; \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip config set global.index-url "${PIP_INDEX_URL}" && \
    python -m pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn"

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" --trusted-host mirrors.nju.edu.cn

COPY requirements.runtime.txt /tmp/requirements.runtime.txt

RUN python -m pip install -r /tmp/requirements.runtime.txt

COPY . .

RUN mkdir -p /app/models_cache /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
