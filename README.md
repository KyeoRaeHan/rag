# RAG기반 서빙 시스템

**RAG (Retrieval-Augmented Generation)** 서빙 체계를 **Kubernetes 환경**에서 안정적으로 구축하는 것을 목표로 합니다.

---

## 개요

- 최신 IT 뉴스 데이터를 임베딩하여 **Chroma 벡터 DB에 저장**
- 사용자 질의에 대해 관련 문서를 DB에서 검색하여 **LLM에 문맥(context)과 함께 질의(question)**
- **FastAPI** 기반 REST API로 `/rag` 엔드포인트 제공
- 모든 구성 요소는 Kubernetes 클러스터에서 배포되며, 노드 배치 및 GPU 리소스를 효율적으로 활용하도록 설계

---

## 구성 요소

| `vectordb-setting` | 마크다운(.md) 문서를 벡터로 임베딩하여 ChromaDB에 저장하는 **일회성 Job** |
| `vectordb-server`  | 저장된 벡터 DB에서 관련 문서를 검색하는 **FastAPI 서버** |
| `llm-api`          | 프롬프트를 입력 받아 답변을 생성하는 **LLM 추론 API** |
| `rag-controller`   | Vector DB에서 context 검색 후, 사용자 질의와 함께 LLM에 전달하여 최종 답변을 제공하는 **중앙 API 컨트롤러** |
| `k8s_manifest.yaml`| 위 구성 요소들을 Kubernetes환경에 배포하는 통합 매니페스트|

---

## 디렉토리 구조
📁 kakao_rag_project/
├── 📁 01_LLM_API/
│   ├── Dockerfile
│   ├── llm_api.py
│   └── requirements.txt
│
├── 📁 02_RAG/
│   ├── 📁 it_tech_news_data/
│   ├── Dockerfile.server
│   ├── Dockerfile.setting
│   ├── requirements.txt
│   ├── vectordb_server.py
│   └── vectordb_setting.py
│
├── 📁 03_DEPLOY/
│   └── k8s_manifest.yaml
│
├── 📁 04_CONTROLLER/
│   ├── Dockerfile
│   ├── rag_controller.py
│   └── requirements.txt
│
└── README.md

---

## 기술 스택
- Python 3.11 (slim 이미지 기반)
- LangChain 0.0.255
- ChromaDB 0.3.29 (DuckDB + Parquet 기반)
- Transformers 4.40.1
- FastAPI + Uvicorn
- Kubernetes

