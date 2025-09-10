# PyTorch + CUDA base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./server.py
COPY scripts/download_models.sh ./scripts/download_models.sh
RUN chmod +x ./scripts/download_models.sh

RUN mkdir -p /models /outputs

ENV MODEL_DIR=/models \
    OUTPUT_DIR=/outputs \
    DEVICE=cuda \
    MAX_WORKERS=1 \
    DOWNLOAD_MODELS=0

EXPOSE 8000

CMD bash -lc 'if [ "$DOWNLOAD_MODELS" = "1" ]; then ./scripts/download_models.sh; fi; \
    uvicorn server:app --host 0.0.0.0 --port 7878'
