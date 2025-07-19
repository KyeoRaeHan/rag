from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import nest_asyncio
import uvicorn
from threading import Thread
import logging

from langchain.vectorstores import Chroma
from chromadb.config import Settings

from vector_db.vectorsdb_setting import KananaEmbeddings


# 설정
app = FastAPI()
logger = logging.getLogger(__name__)

# Colab환경에서 사용
# nest_asyncio.apply()


# 모델 및 임베딩 로드
LLM_MODEL_NAME = "nlpai-lab/kullm-polyglot-5.8b-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto").to(device)

embedder = KananaEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    device=device
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db",
        anonymized_telemetry=False # 익명 사용 정보 외부 전송 비활성화. 필수!
    )
)


# Retriever 객체로 변환
retriever = vectordb.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.8})


# 요청/응답 스키마
class GenerateRequest(BaseModel):
    query: str

class GenerateResponse(BaseModel):
    answer: str

class PromptRequest(BaseModel):
    query: str

class PromptResponse(BaseModel):
    answer: str


# 엔드포인트
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}

@app.post("/generate", summary="Text generation")
def generate(request: GenerateRequest):

    query = request.query

    prompt = f"""문서 내용을 참고하여 질문에 답해보자.

    질문: {query}
    답변:
    """

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        inputs.pop("token_type_ids", None)

        # 추론 시 gradient 계산 비활성화 (메모리 절약)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=200)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return GenerateResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/rag", summary="Rag")
def rag_query(request: PromptRequest):

    query = request.query
    
    # 질의(query)에 유사한 문서 3개 검색
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""문서 내용을 참고하여 질문에 답해보자.

    문서: {context}
    질문: {query}
    답변:
    """

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        inputs.pop("token_type_ids", None)

        # 추론 시 gradient 계산 비활성화 (메모리 절약)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=200)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return PromptResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rag failed: {str(e)}")
