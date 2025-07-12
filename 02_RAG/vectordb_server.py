import os
import torch
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from chromadb.config import Settings

from vectordb_setting import KananaEmbeddings

# 환경변수
PERSIST_DIR = os.getenv("PERSIST_DIR", "/app/chroma_db")
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/e5-small-v2")
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "1") == "1" and torch.cuda.is_available() else "cpu")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("vectordb-server")

# FastAPI 초기화
app = FastAPI(title="vectordb server")

# 임베딩 모델 및 벡터 DB 로드
embedder = KananaEmbeddings(
    model_name=MODEL_NAME,
    device=DEVICE
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedder,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR,
        anonymized_telemetry=False
    )
)

# 요청 스키마
class QueryRequest(BaseModel):
    query: str

# 엔드포인트
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve_docs(request: QueryRequest):
    docs = vectordb.similarity_search(request.query, k=3)
    return {"docs": [{"text": doc.page_content} for doc in docs]}
