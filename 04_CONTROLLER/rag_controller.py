import os
import logging
from typing import List, Optional
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,RetrievalQA
from langchain.schema import BaseRetriever, Document
from langchain.llms.base import LLM


# 환경 변수
VECTORDB_ENDPOINT = os.getenv("VECTORDB_ENDPOINT", "").rstrip("/")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "").rstrip("/")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

if not VECTORDB_ENDPOINT or not LLM_ENDPOINT:
    raise RuntimeError("VECTORDB_ENDPOINT / LLM_ENDPOINT 가 필요합니다.")

# 로깅
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag-controller")


# LangChain 커스텀 구성 요소
class RemoteRetriever(BaseRetriever):

    """VectorDB REST API → LangChain Retriever"""
    def __init__(self, endpoint: str, client: httpx.AsyncClient, top_k: int = 3):
        super().__init__()
        self.endpoint = endpoint
        self.client = client
        self.top_k = top_k

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        resp = await self.client.post(self.endpoint, json={"query": query, "k": self.top_k})
        resp.raise_for_status()
        data = resp.json()
        return [
            Document(page_content=ctx["text"], metadata={"source": ctx.get("source", "")})
            for ctx in data.get("contexts", [])
        ]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            return asyncio.run(self._aget_relevant_documents(query))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(self._aget_relevant_documents(query))

class RemoteLLM(LLM):

    """LLM REST API → LangChain LLM"""
    def __init__(self, endpoint: str, client: httpx.AsyncClient):
        self.endpoint = endpoint
        self.client = client

    @property
    def _llm_type(self) -> str:
        return "remote_llm"

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        body = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_new_tokens", MAX_NEW_TOKENS),
            "temperature": TEMPERATURE,
            "stop": stop,
        }
        resp = await self.client.post(self.endpoint, json=body)
        resp.raise_for_status()
        return resp.json().get("generated_text", "")

# FastAPI + LangChain 초기화
app = FastAPI(title="RAG Controller")

@app.on_event("startup")
async def on_startup():
    client = httpx.AsyncClient(timeout=httpx.Timeout(30))
    app.state.client = client

    # 프롬프트 템플릿 정의
    template = """You are a helpful assistant. Based on the following context, answer the question concisely.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # LangChain 구성
    retriever = RemoteRetriever(VECTORDB_ENDPOINT, client)
    llm = RemoteLLM(LLM_ENDPOINT, client)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",             #문서들을 한번에 프롬프트에 넣음
        retriever=retriever,
        return_source_documents=True,  # 문맥 함께 반환
        chain_type_kwargs={"prompt": prompt},
    )

    app.state.rag_chain = rag_chain
    log.info("✅ LangChain RAG pipeline ready")

@app.on_event("shutdown")
async def on_shutdown():
    await app.state.client.aclose()

# 요청 스키마
class RAGRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    top_k: Optional[int] = Field(3, ge=1, le=20)
    stop: Optional[List[str]] = None

# 응답 스키마
class ContextChunk(BaseModel):
    text: str
    source: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    contexts: List[ContextChunk]

# 엔드포인트
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rag", response_model=RAGResponse)
async def rag(req: RAGRequest):
    chain: RetrievalQA = app.state.rag_chain
    retriever: RemoteRetriever = chain.retriever

    retriever.top_k = req.top_k or 3

    # LangChain RetrievalQA 는 동기 함수 → threadpool 로 오프로드
    result = await run_in_threadpool(chain, {"query": req.query, "max_new_tokens": req.max_new_tokens or MAX_NEW_TOKENS})
    answer = result["result"]
    docs = result["source_documents"]

    contexts = [ContextChunk(text=d.page_content, source=d.metadata.get("source")) for d in docs]
    return RAGResponse(answer=answer, contexts=contexts)
