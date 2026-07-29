FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/repo

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      git \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip==25.1.1" && \
    python -m pip install \
      "numpy==1.26.4" \
      "matplotlib==3.10.3" \
      "packaging==25.0" \
      "prettytable==3.16.0" \
      "scipy==1.15.3" \
      "mmcv-lite==2.1.0" \
      "mmengine==0.10.7" \
      "clearml==2.1.10" \
      "ftfy==6.3.1" \
      "regex==2024.11.6"

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

WORKDIR /workspace/repo
