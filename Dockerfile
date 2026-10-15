FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HOME=/home/mediko \
    PADDLE_PDX_CACHE_HOME=/home/mediko/.paddlex \
    MEDIKO_CONFIG=/app/config/default.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/mediko --shell /bin/sh --uid 1000 mediko

WORKDIR /app

COPY requirements.txt .
# CPU wheel from PyPI. OCR model weights are not downloaded during the
# image build; the first scanned page stores them in the paddlex volume.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import paddle; print(paddle.__version__)"

COPY src ./src
COPY app ./app
COPY config ./config
COPY .streamlit ./.streamlit
COPY docker/entrypoint.sh /entrypoint.sh

RUN sed -i "s/\r$//" /entrypoint.sh \
    && chmod 755 /entrypoint.sh \
    && mkdir -p /home/mediko/.paddlex \
    && chown -R mediko:mediko /app /home/mediko

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py"]
