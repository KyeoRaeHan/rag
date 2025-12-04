from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from langchain.vectorstores import Chroma
from chromadb.config import Settings

from vector_db.vectorsdb_setting import KananaEmbeddings


# 설정
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 모델 설정 상수
LLM_MODEL_NAME = "nlpai-lab/kullm-polyglot-5.8b-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 생성 파라미터 설정
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 200
}

# 검색 파라미터 설정
RETRIEVER_CONFIG = {
    "k": 3,
    "score_threshold": 0.8
}

# 토크나이저 설정
TOKENIZER_CONFIG = {
    "return_tensors": "pt",
    "truncation": True,
    "max_length": 2048
}


# 모델 및 임베딩 로드
logger.info(f"Loading LLM model: {LLM_MODEL_NAME} on {device}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
).to(device)

logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedder = KananaEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    device=device
)

logger.info("Initializing ChromaDB")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db",
        anonymized_telemetry=False  # 익명 사용 정보 외부 전송 비활성화. 필수!
    )
)

retriever = vectordb.as_retriever(search_kwargs=RETRIEVER_CONFIG)


# 요청/응답 스키마
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


# 유틸리티 함수
def create_prompt(query: str, context: str = None) -> str:
    """프롬프트 생성 함수"""
    if context:
        return f"""문서 내용을 참고하여 질문에 답해보자.

문서: {context}
질문: {query}
답변:
"""
    else:
        return f"""문서 내용을 참고하여 질문에 답해보자.

질문: {query}
답변:
"""


def generate_text(prompt: str) -> str:
    """텍스트 생성 공통 함수"""
    try:
        # 토크나이징
        inputs = tokenizer(prompt, **TOKENIZER_CONFIG).to(device)
        inputs.pop("token_type_ids", None)

        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **GENERATION_CONFIG)

        # 디코딩
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # 프롬프트 부분 제거 (원본 프롬프트 제외하고 답변만 추출)
        if "답변:" in answer:
            answer = answer.split("답변:")[-1].strip()
        
        return answer

    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        raise HTTPException(
            status_code=503, 
            detail="Model inference failed: Out of memory"
        )
    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Text generation failed: {str(e)}"
        )


# 엔드포인트
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "device": device}


@app.post("/generate", summary="Text generation")
def generate(request: QueryRequest):
    """문서 검색 없이 질문만으로 텍스트 생성"""
    logger.info(f"Generate request: {request.query[:50]}...")
    
    prompt = create_prompt(request.query)
    answer = generate_text(prompt)
    
    logger.info("Generation completed successfully")
    return QueryResponse(answer=answer)


@app.post("/rag", summary="RAG (Retrieval Augmented Generation)")
def rag_query(request: QueryRequest):
    """벡터 DB에서 관련 문서를 검색하여 컨텍스트로 사용한 텍스트 생성"""
    logger.info(f"RAG request: {request.query[:50]}...")
    
    try:
        # 문서 검색
        docs = retriever.get_relevant_documents(request.query)
        
        if not docs:
            logger.warning("No relevant documents found, proceeding without context")
            # 문서를 찾지 못해도 LLM이 일반 지식으로 답변하도록 진행
            prompt = create_prompt(request.query)
        else:
            # 컨텍스트 생성
            context = "\n\n".join(doc.page_content for doc in docs)
            logger.info(f"Retrieved {len(docs)} documents")
            # 프롬프트 생성
            prompt = create_prompt(request.query, context)
        
        # 텍스트 생성
        answer = generate_text(prompt)
        
        logger.info("RAG completed successfully")
        return QueryResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG failed: {str(e)}"
        )
