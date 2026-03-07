"""
HDFC MF RAG — FastAPI Server
=============================
Phase 4: REST API wrapper around rag_backend.query_rag()

Endpoints:
  GET  /health          — liveness probe
  GET  /funds           — list of supported fund IDs + display names
  POST /ask             — main RAG endpoint

Run locally:
  uvicorn backend.api:app --reload --port 8000

Production (Render/Railway):
  uvicorn backend.api:app --host 0.0.0.0 --port $PORT
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so we can import backend.rag_backend
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.rag_backend import query_rag, FUND_DISPLAY_NAMES

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HDFC MF RAG API",
    description="Hyper-accurate Q&A over HDFC Mutual Fund documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow any Vercel preview URL + localhost during development
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001,https://*.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to ALLOWED_ORIGINS before prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class AskRequest(BaseModel):
    fund_id: str = Field(
        ...,
        description="One of: hdfc_large_cap | hdfc_elss | hdfc_flexi_cap",
        example="hdfc_large_cap",
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language question about the fund",
        example="What is the expense ratio?",
    )


class AskResponse(BaseModel):
    answer:       str
    source_url:   str
    last_updated: str
    fund_name:    str
    refused:      bool
    refusal_type: str  # 'pii' | 'advice' | 'no_context' | ''


class FundInfo(BaseModel):
    fund_id:    str
    fund_name:  str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe — used by Render, Railway, k8s, UptimeRobot etc."""
    return {"status": "ok"}


@app.get("/funds", response_model=list[FundInfo], tags=["Funds"])
async def list_funds():
    """Return the list of supported fund IDs and their display names."""
    return [
        FundInfo(fund_id=fid, fund_name=name)
        for fid, name in FUND_DISPLAY_NAMES.items()
    ]


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """
    Query the RAG pipeline for a specific HDFC fund.

    - Guardrails: PII → auto-refused; investment advice → auto-refused
    - Confidence: if question is off-topic → refused with no_context
    - Citation: every successful answer includes a direct PDF source link
    """
    if request.fund_id not in FUND_DISPLAY_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown fund_id '{request.fund_id}'. "
                   f"Valid options: {list(FUND_DISPLAY_NAMES.keys())}",
        )

    result = query_rag(fund_id=request.fund_id, question=request.question)
    return AskResponse(**result)


# ─── Dev entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
