# rag_controller Deployment, rag_controller:latest 도커 이미지
FROM python:3.11-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 필요한 파일만 복사 (vector_db 모듈 import를 위해 상위 구조 유지)
COPY rag_controller/requirements.txt requirements.txt
COPY rag_controller/rag_controller.py rag_controller.py
COPY vector_db/vectordb_setting.py vector_db/vectordb_setting.py

# 설치 중 OOM 방지 및 tokenizer 관련 에러 방지 옵션 설정
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HUB_CACHE=/app/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "rag_controller:app", "--host", "0.0.0.0", "--port", "8000"]
