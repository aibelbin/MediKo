FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU wheel by default (avoids ~2 GB of CUDA libs). For an NVIDIA GPU build with
# docker-compose.gpu.yml, which sets TORCH_INDEX to the CUDA wheel index.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url ${TORCH_INDEX} torch==2.14.1 \
 && pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY entrypoint.sh /entrypoint.sh
RUN sed -i "s/\r$//" /entrypoint.sh && chmod 755 /entrypoint.sh \
 && python -m unittest discover -s tests -v

# The dataset and model weights are downloaded on first run into these volumes.
VOLUME ["/app/data", "/app/.hf_cache", "/app/outputs"]
ENTRYPOINT ["/entrypoint.sh"]
