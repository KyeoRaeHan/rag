import argparse
import logging
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma

from chromadb.config import Settings


# 로깅
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("vectordb-setting")

# 임베딩 클래스
class KananaEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device).eval()

    def _encode(self, texts: List[str], chunk_size: int = 32) -> List[List[float]]:
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
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

# 벡터DB 구축
def build_chroma(
        data_path: str,
        persist_dir: str,
        batch_size: int,
        persist_every: int,
        preview: int,
        model_name: str,
        device: str):

    log.info(f"▶ 시작: data_path={data_path} | persist_dir={persist_dir}")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ]
    )
    embedder = KananaEmbeddings(model_name, device)

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder,
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )

    loader = DirectoryLoader(data_path, glob="**/*.md")
    docs = loader.load()

    log.info(f"총 파일 수: {len(docs)}")

    batch: List[Document] = []
    batch_count = 0
    total_chunks = 0

    for doc in docs:
        source = doc.metadata.get("source", "")
        split_docs: List[Document] = splitter.split_text(doc.page_content)

        for idx, chunk in enumerate(split_docs):
            chunk.metadata["source"] = source  # 원본 파일 이름
            chunk.metadata["chunk_id"] = idx   # 인덱싱
            batch.append(chunk)
            total_chunks += 1

            if len(batch) >= batch_size:
                batch_count += 1
                _flush_batch(vectordb, batch, batch_count)
                batch.clear()
                if persist_every > 0 and batch_count % persist_every == 0:
                    vectordb.persist()

    if batch:
        batch_count += 1
        _flush_batch(vectordb, batch, batch_count)

    vectordb.persist()
    log.info("최종 저장")
    log.info(f"완료 | 총 청크 수: {total_chunks} | 배치 수: {batch_count}")

    if preview > 0:
        sample = vectordb.get(include=["documents"], limit=preview)
        for i, doc in enumerate(sample["documents"]):
            clean_doc = doc[:50].replace("\n", " ")
            log.info(f"Preview #{i+1} | chunk='{clean_doc}...'")

    return vectordb.as_retriever(search_kwargs={"k": 3})

# 배치 flush
def _flush_batch(vectordb: Chroma, docs_batch: List[Document], batch_idx: int):

    log.info(f"▶ Batch {batch_idx} | size={len(docs_batch)}")
    vectordb.add_documents(docs_batch)

# main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vectordb builder")
    parser.add_argument("--data-path", default=os.getenv("DATA_PATH", "./it_tech_news_data"))
    parser.add_argument("--persist-dir", default=os.getenv("PERSIST_DIR", "./chroma_db"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 64)))
    parser.add_argument("--persist-every", type=int, default=int(os.getenv("PERSIST_EVERY", 3)))
    parser.add_argument("--preview", type=int, default=int(os.getenv("PREVIEW", 2)))
    parser.add_argument("--model-name", default="intfloat/e5-small-v2")

    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    build_chroma(
        data_path=args.data_path,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        persist_every=args.persist_every,
        preview=args.preview,
        model_name=args.model_name,
        device=device
    )