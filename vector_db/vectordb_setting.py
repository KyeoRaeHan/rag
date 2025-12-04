import argparse
import logging
import os
from typing import List, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModel

from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma

from chromadb.config import Settings


# 로깅
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("vectordb-setting")


def get_device() -> str:
    """GPU 사용 가능 여부에 따라 디바이스 반환"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# 임베딩 클래스
class KananaEmbeddings(Embeddings):
    """LangChain Embeddings 인터페이스를 구현한 커스텀 임베딩 클래스"""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Args:
            model_name: Hugging Face 모델 이름
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device).eval()

    def _encode(self, texts: List[str], chunk_size: int = 32) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            chunk_size: 배치 크기
            
        Returns:
            임베딩 벡터 리스트
        """
        all_embeddings = []

        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:i + chunk_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                # Mean Pooling
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                count = torch.clamp(mask.sum(dim=1), min=1e-9)
                embeddings = summed / count
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩"""
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 텍스트를 임베딩"""
        return self._encode([text])[0]

# 문서 로딩
def load_documents(data_path: str) -> List[Document]:
    """
    디렉토리에서 마크다운 문서 로딩
    
    Args:
        data_path: 문서가 있는 디렉토리 경로
        
    Returns:
        로드된 문서 리스트
        
    Raises:
        Exception: 문서 로딩 실패 시
    """
    try:
        loader = DirectoryLoader(data_path, glob="**/*.md")
        docs = loader.load()
        
        if not docs:
            log.warning(f"No documents found in {data_path}")
            return []
            
        log.info(f"총 파일 수: {len(docs)}")
        return docs
    except Exception as e:
        log.error(f"Failed to load documents from {data_path}: {e}")
        raise


# 문서 분할
def split_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    문서를 헤더 기준과 청크 사이즈로 분할
    
    Args:
        docs: 분할할 문서 리스트
        chunk_size: 청크 최대 크기 (문자 수)
        chunk_overlap: 청크 간 겹치는 문자 수
        
    Returns:
        분할된 청크 리스트
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("###", "h3"), ("##", "h2"), ("#", "h1")]
    )
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n", " "]
    )
    
    all_chunks = []
    
    for doc in docs:
        source = doc.metadata.get("source", "")
        header_sections = header_splitter.split_text(doc.page_content)
        
        for section in header_sections:
            section_doc = Document(page_content=str(section), metadata={"source": source})
            split_docs = char_splitter.split_documents([section_doc])
            
            for idx, chunk in enumerate(split_docs):
                chunk.metadata["chunk_id"] = idx
                all_chunks.append(chunk)
    
    return all_chunks


# 벡터 DB 초기화
def initialize_vectordb(
    persist_dir: str,
    embedder: KananaEmbeddings
) -> Chroma:
    """
    Chroma 벡터 DB 초기화
    
    Args:
        persist_dir: 벡터 DB 저장 디렉토리
        embedder: 임베딩 함수
        
    Returns:
        초기화된 Chroma 벡터 DB 객체
    """
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder,
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",  # 별도 DB서버 불필요, 배포 용이
            anonymized_telemetry=False  # 익명 사용 정보 외부 전송 비활성화. 필수!
        )
    )


# 배치 처리
def process_batches(
    vectordb: Chroma,
    chunks: List[Document],
    batch_size: int,
    persist_every: int
) -> tuple[int, int]:
    """
    청크들을 배치 단위로 처리하여 벡터 DB에 추가
    
    Args:
        vectordb: Chroma 벡터 DB 객체
        chunks: 처리할 청크 리스트
        batch_size: 배치 크기
        persist_every: 디스크 저장 주기 (N배치마다)
        
    Returns:
        (총 청크 수, 총 배치 수) 튜플
        
    Raises:
        Exception: 배치 처리 실패 시
    """
    batch: List[Document] = []
    batch_count = 0
    total_chunks = 0
    
    for chunk in chunks:
        batch.append(chunk)
        total_chunks += 1
        
        if len(batch) >= batch_size:
            batch_count += 1
            try:
                _flush_batch(vectordb, batch, batch_count)  # 메모리에 저장
                batch.clear()
                if persist_every > 0 and batch_count % persist_every == 0:
                    vectordb.persist()  # 디스크에 저장
            except Exception as e:
                log.error(f"Failed to process batch {batch_count}: {e}")
                raise
    
    # 남은 배치 처리
    if batch:
        batch_count += 1
        try:
            _flush_batch(vectordb, batch, batch_count)
        except Exception as e:
            log.error(f"Failed to flush final batch: {e}")
            raise
    
    return total_chunks, batch_count


# 미리보기 검색
def preview_search(
    vectordb: Chroma,
    query: str,
    k: int
) -> None:
    """
    벡터 DB에서 검색 미리보기
    
    Args:
        vectordb: Chroma 벡터 DB 객체
        query: 검색 쿼리
        k: 반환할 문서 수
    """
    if k <= 0:
        return
        
    try:
        sample = vectordb.similarity_search(query, k=k)
        for i, doc in enumerate(sample):
            clean_doc = doc.page_content[:80].replace("\n", " ")
            log.info(f" Preview #{i+1} | '{clean_doc}...'")
    except Exception as e:
        log.warning(f"Failed to perform preview search: {e}")


# 벡터DB 구축
def build_chroma(
        data_path: str,
        persist_dir: str,
        batch_size: int,
        persist_every: int,
        preview: int,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        device: str,
        preview_query: str = "MCP에 대해 설명해줘",
        retriever_k: int = 3,
        retriever_score_threshold: float = 0.8) -> Optional[Any]:
    """
    벡터 DB 구축 메인 함수
    
    Args:
        data_path: 문서 데이터 경로
        persist_dir: 벡터 DB 저장 디렉토리
        batch_size: 배치 크기
        persist_every: 디스크 저장 주기
        preview: 미리보기 결과 수
        model_name: 임베딩 모델 이름
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹치는 문자 수
        device: 사용할 디바이스
        preview_query: 미리보기 검색 쿼리
        retriever_k: Retriever가 반환할 문서 수
        retriever_score_threshold: Retriever 최소 유사도 점수
        
    Returns:
        LangChain Retriever 객체 또는 None (문서 없을 시)
    """
    log.info(f"시작: data_path={data_path} | persist_dir={persist_dir}")

    # 임베딩 모델 초기화
    embedder = KananaEmbeddings(model_name, device)

    # 벡터 DB 초기화
    vectordb = initialize_vectordb(persist_dir, embedder)

    # 문서 로딩
    docs = load_documents(data_path)
    if not docs:
        return None

    # 문서 분할
    chunks = split_documents(docs, chunk_size, chunk_overlap)

    # 배치 처리
    total_chunks, batch_count = process_batches(
        vectordb, chunks, batch_size, persist_every
    )

    # 최종 저장
    try:
        vectordb.persist()
        log.info("최종 저장 완료")
        log.info(f"총 청크 수: {total_chunks} | 총 배치 수: {batch_count}")
    except Exception as e:
        log.error(f"Failed to persist vector database: {e}")
        raise

    # Preview
    preview_search(vectordb, preview_query, preview)

    return vectordb.as_retriever(
        search_kwargs={"k": retriever_k, "score_threshold": retriever_score_threshold}
    )

# 배치 flush
def _flush_batch(vectordb: Chroma, docs_batch: List[Document], batch_idx: int) -> None:
    """배치 단위로 벡터 DB에 문서 추가"""
    log.info(f"Batch {batch_idx} | size={len(docs_batch)}")
    try:
        vectordb._collection.add(
            documents=[d.page_content for d in docs_batch],
            metadatas=[d.metadata for d in docs_batch],
            ids=[f"batch{batch_idx}-{i}" for i in range(len(docs_batch))]
        )
    except Exception as e:
        log.error(f"Failed to add batch {batch_idx} to vector database: {e}")
        raise

# main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vectordb-setting")
    parser.add_argument("--data-path", default=os.getenv("DATA_PATH", "./it_tech_news_data"))
    parser.add_argument("--persist-dir", default=os.getenv("PERSIST_DIR", "./chroma_db"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 32)))
    parser.add_argument("--persist-every", type=int, default=int(os.getenv("PERSIST_EVERY", 3)))
    parser.add_argument("--preview", type=int, default=int(os.getenv("PREVIEW", 2)))
    parser.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--chunk-size", type=int, default=70, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap characters between chunks")
    parser.add_argument("--preview-query", type=str, default="MCP에 대해 설명해줘", help="Preview search query")
    parser.add_argument("--retriever-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--retriever-score-threshold", type=float, default=0.8, help="Minimum similarity score threshold")

    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    build_chroma(
        data_path=args.data_path,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        persist_every=args.persist_every,
        preview=args.preview,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        device=device,
        preview_query=args.preview_query,
        retriever_k=args.retriever_k,
        retriever_score_threshold=args.retriever_score_threshold
    )
