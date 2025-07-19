# RAG기반 서빙 시스템

**RAG(Retrieval-Augmented Generation)** 서빙 체계를 **Kubernetes 환경**에 안정적으로 구축하는 것을 목표로 합니다.

- 최신 IT 뉴스 데이터를 임베딩하여 **Chroma 벡터 DB에 저장**
- 사용자 질의에 대한 관련 문서를 DB에서 검색하여 **LLM에 문맥(context)과 함께 질의(question)**
- **FastAPI** 기반 REST API로  `/generate` `/rag` 엔드포인트 제공
- 모든 구성 요소는 Kubernetes 클러스터에 배포되며, 노드 배치 및 GPU 리소스를 효율적으로 활용하도록 설계
- 외부 통신 가능하도록 NodePort로 서비스 설정

## 구성 요소
- `vectordb-setting` 마크다운(.md) 문서를 벡터로 임베딩하여 ChromaDB에 저장
- `rag-controller`   Vector DB에서 context 검색 후, 사용자 질의와 함께 LLM에 전달하여 최종 답변을 제공하는 컨트롤러
- `k8s_manifest.yaml`위 구성 요소들을 Kubernetes 환경에 배포하는 통합 매니페스트 파일

## 프로젝트 구조도
```
       ┌────────────┐
       │   Client   │
       └─────┬──────┘
             │
             ▼
┌───────────────────────────────┐
│     FastAPI /rag Endpoint     │
└──────────────┬────────────────┘
               │
               ▼
        ┌─────────────┐
        │  Retriever  │  ← LangChain
        └─────┬───────┘
              │
              ▼
     ┌─────────────────────┐
     │  Vector DB (Chroma) │
     └────────┬────────────┘
              │  ← Relevant Documents (Context) 전달
              ▼
┌────────────────────────────┐
│   Prompt Builder           │
│ (query + context → prompt) │
└────────────┬───────────────┘
             ▼
      ┌───────────────┐
      │     LLM       │ 
      └──────┬────────┘
             ▼
      ┌───────────────┐
      │   Response    │  ← answer 생성
      └──────┬────────┘
             ▼
      ┌───────────────┐
      │   Client UI   │
      └───────────────┘
```

## 기술 스택
- Python 3.11 (slim 이미지 기반)
- LangChain 0.0.255
- ChromaDB 0.3.29
- Transformers 4.40.1
- pydantic 1.10.13
- huggingface-hub 0.23.0
- torch 2.1.2
- FastAPI + Uvicorn
- Kubernetes

