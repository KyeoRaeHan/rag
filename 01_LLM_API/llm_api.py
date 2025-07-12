import os
import logging
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# 환경 변수
MODEL_NAME: str = os.getenv("MODEL_NAME", "kakaocorp/kanana-nano-2.1b-instruct")
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.5"))
DEFAULT_TOP_P: float = float(os.getenv("TOP_P", "0.95"))

DEVICE: int = 0 if torch.cuda.is_available() else -1

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm_api")

# 모델 로드 (Hugging Face Pipeline)
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=DEVICE,
    trust_remote_code=True,
    batch_size=MAX_BATCH_SIZE,
)
logger.info("Model loaded and ready")

# FastAPI 앱 & 스키마
app = FastAPI(
    title="Kanana LLM API",
    description="Generates text with kakaocorp/kanana-nano-2.1b-instruct",
)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="프롬프트 / 사용자 질문")
    max_new_tokens: int = Field(
        DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096, description="생성할 최대 토큰 수"
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="샘플링 temperature"
    )
    top_p: float = Field(
        DEFAULT_TOP_P, ge=0.0, le=1.0, description="nucleus sampling top-p"
    )
    stop: Optional[List[str]] = Field(
        None, description="stop sequence 목록"
    )


class GenerateResponse(BaseModel):
    generated_text: str


# 엔드포인트
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse, summary="Text generation")
def generate(req: GenerateRequest):

    # 프롬프트를 받아 Kanana 2.1B Instruct 모델로 텍스트 생성
    try:
        outputs = generator(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            num_return_sequences=1,
            eos_token_id=None,  # → 모델 default
        )
        text: str = outputs[0]["generated_text"]

        # stop sequence 처리
        if req.stop:
            for stop_seq in req.stop:
                idx = text.find(stop_seq)
                if idx != -1:
                    text = text[:idx]
                    break

        return GenerateResponse(generated_text=text)

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))