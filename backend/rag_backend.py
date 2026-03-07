"""
HDFC MF RAG — Backend Query Engine
====================================
Phase 3 of the RAG pipeline.

Flow for each query:
  1. Embed the question using the same local sentence-transformers model.
  2. Query ChromaDB with a fund_id filter → top-K chunks.
  3. Confidence threshold: if best distance > threshold → politely refuse.
  4. Refusal guard: if question asks for investment advice / contains PII patterns → refuse.
  5. Build a context-grounded prompt and call Gemini 1.5 Flash.
  6. Return { answer, source_url, last_updated, fund_id }.

Usage (standalone test):
  python backend/rag_backend.py

Usage (as a module):
  from backend.rag_backend import query_rag
  result = query_rag(fund_id="hdfc_large_cap", question="What is the expense ratio?")
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types as genai_types

# ─── Paths & env ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT        = os.getenv("SYSTEM_PROMPT", "")
CHROMA_DB_PATH       = os.getenv("CHROMA_DB_PATH", str(ROOT_DIR / "vector_db"))
CHROMA_COLLECTION    = os.getenv("CHROMA_COLLECTION_NAME", "hdfc_mf_docs")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K                = int(os.getenv("TOP_K_RESULTS", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "1.4"))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Fund display names ────────────────────────────────────────────────────────
FUND_DISPLAY_NAMES = {
    "hdfc_large_cap": "HDFC Large Cap Fund",
    "hdfc_elss":      "HDFC ELSS Tax Saver",
    "hdfc_flexi_cap": "HDFC Flexi Cap Fund",
}

# ─── Refusal patterns ─────────────────────────────────────────────────────────
# Investment advice keywords
ADVICE_PATTERNS = re.compile(
    r"\b(should\s+i|shall\s+i|recommend|advise|buy|sell|invest\s+in|safe\s+to\s+invest"
    r"|is\s+it\s+good|is\s+it\s+worth|better\s+fund|best\s+fund|top\s+fund"
    r"|which\s+fund|portfolio|returns\s+predict|future\s+return|will\s+it\s+grow"
    r"|can\s+i\s+make\s+money|profit\s+from)\b",
    re.IGNORECASE,
)

# PII patterns — refuse if detected in the question
PII_PATTERNS = re.compile(
    r"\b([A-Z]{5}[0-9]{4}[A-Z]"          # PAN
    r"|[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}"  # Aadhaar (partial)
    r"|[0-9]{16,19}"                       # Card / account number
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # Email
    r"|\b[6-9][0-9]{9}\b"                  # Indian mobile
    r")\b",
    re.IGNORECASE,
)

REFUSAL_ADVICE = (
    "I can only share factual information from HDFC Mutual Fund documents. "
    "For investment advice, please consult a SEBI-registered financial adviser "
    "or visit https://www.hdfcfund.com."
)

REFUSAL_PII = (
    "For your privacy, I don't accept or store personal information like PAN, "
    "Aadhaar, account numbers, emails, or phone numbers. "
    "Please re-phrase your question without personal details."
)

REFUSAL_NO_CONTEXT = (
    "I could not find relevant information in the HDFC Mutual Fund documents "
    "for your question. Please try rephrasing, or visit https://www.hdfcfund.com "
    "for complete fund details."
)


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB client (singleton-ish — cached at module level)
# ═══════════════════════════════════════════════════════════════════════════════

_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        db_path = str(ROOT_DIR / "vector_db") if CHROMA_DB_PATH.startswith(".") else CHROMA_DB_PATH
        
        # We need to tell Chroma to not try to acquire write locks if deployed
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=False)
        )
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB ready — {_collection.count()} vectors.")
    return _collection


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini client
# ═══════════════════════════════════════════════════════════════════════════════

def _get_gemini_client():
    """Return a configured google-genai client."""
    return genai.Client(api_key=GEMINI_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrail checks
# ═══════════════════════════════════════════════════════════════════════════════

def _check_pii(question: str) -> Optional[str]:
    """Returns refusal message if PII detected, else None."""
    if PII_PATTERNS.search(question):
        return REFUSAL_PII
    return None

def _check_advice(question: str) -> Optional[str]:
    """Returns refusal message if investment-advice intent detected, else None."""
    if ADVICE_PATTERNS.search(question):
        return REFUSAL_ADVICE
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Core RAG function
# ═══════════════════════════════════════════════════════════════════════════════

def query_rag(fund_id: str, question: str) -> dict:
    """
    Main entry point for the RAG pipeline.

    Parameters
    ----------
    fund_id  : One of 'hdfc_large_cap', 'hdfc_elss', 'hdfc_flexi_cap'
    question : The user's natural-language question

    Returns
    -------
    dict with keys:
        answer       : str  — the response text (may be a refusal message)
        source_url   : str  — direct PDF link (empty on refusal)
        last_updated : str  — ISO timestamp of the indexed PDF (empty on refusal)
        fund_name    : str  — human-readable fund name
        refused      : bool — True if the query was refused
        refusal_type : str  — 'pii' | 'advice' | 'no_context' | '' 
    """
    fund_name = FUND_DISPLAY_NAMES.get(fund_id, fund_id)

    # ── Guard 1: PII check ───────────────────────────────────────────────────
    pii_refusal = _check_pii(question)
    if pii_refusal:
        log.info(f"[REFUSAL:pii] {fund_id} — {question[:60]}")
        return _refusal_response(fund_name, pii_refusal, "pii")

    # ── Guard 2: Investment-advice check ────────────────────────────────────
    advice_refusal = _check_advice(question)
    if advice_refusal:
        log.info(f"[REFUSAL:advice] {fund_id} — {question[:60]}")
        return _refusal_response(fund_name, advice_refusal, "advice")

    # ── Step 1: Retrieve relevant chunks from ChromaDB ──────────────────────
    collection = _get_collection()
    try:
        results = collection.query(
            query_texts=[question],
            n_results=TOP_K,
            where={"fund_id": fund_id},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.error(f"ChromaDB query error: {e}")
        return _refusal_response(fund_name, REFUSAL_NO_CONTEXT, "no_context")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return _refusal_response(fund_name, REFUSAL_NO_CONTEXT, "no_context")

    # ── Guard 3: Confidence threshold ────────────────────────────────────────
    best_distance = distances[0] if distances else 999.0
    log.info(f"[RETRIEVAL] Best distance: {best_distance:.4f} (threshold: {CONFIDENCE_THRESHOLD})")
    if best_distance > CONFIDENCE_THRESHOLD:
        log.info(f"[REFUSAL:no_context] distance too high ({best_distance:.4f})")
        return _refusal_response(fund_name, REFUSAL_NO_CONTEXT, "no_context")

    # ── Step 2: Small-to-Big Context Expansion ──────────────────────────────
    source_url   = metadatas[0].get("source_url", "") if metadatas else ""
    last_updated = metadatas[0].get("last_updated", "") if metadatas else ""

    expanded_docs = []
    seen_chunks = set()
    
    for i, meta in enumerate(metadatas):
        doc_type = meta.get("doc_type")
        c_idx = meta.get("chunk_index")
        
        if doc_type and c_idx is not None:
            try:
                neighbors = collection.get(
                    where={
                        "$and": [
                            {"fund_id": {"$eq": fund_id}},
                            {"doc_type": {"$eq": doc_type}},
                            {"chunk_index": {"$in": [int(c_idx) - 1, int(c_idx), int(c_idx) + 1]}}
                        ]
                    },
                    include=["documents", "metadatas"]
                )
                for n_doc, n_meta in zip(neighbors["documents"], neighbors["metadatas"]):
                    n_id = (n_meta.get("doc_type"), n_meta.get("chunk_index"))
                    if n_id not in seen_chunks:
                        seen_chunks.add(n_id)
                        expanded_docs.append((n_meta, n_doc))
            except Exception as e:
                log.warning(f"Error expanding chunks for {doc_type} index {c_idx}: {e}")
                expanded_docs.append((meta, documents[i]))
        else:
            expanded_docs.append((meta, documents[i]))

    # Sort sequentially so LLM reads it natively like a document
    expanded_docs.sort(key=lambda x: (x[0].get("doc_type", ""), x[0].get("chunk_index", 0)))

    context_text = "\n\n---\n\n".join(
        f"[Chunk {m.get('chunk_index', 'N/A')} | {m.get('doc_type','').upper()}]\n{doc}"
        for m, doc in expanded_docs
    )

    prompt = (
        f"Fund: {fund_name}\n\n"
        f"Context from official HDFC documents:\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer (3 sentences max, strictly from the context above):"
    )

    # ── Step 3: Call Gemini ──────────────────────────────────────────────────
    system_instr = SYSTEM_PROMPT or (
        "You are a strictly factual assistant for HDFC Mutual Fund documents. "
        "Answer in 3 sentences or fewer using only the provided context. "
        "Never give investment advice."
    )
    full_prompt = f"{system_instr}\n\n{prompt}"
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        answer = response.text.strip()
        log.info(f"[GEMINI] Answer generated ({len(answer)} chars).")
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return _refusal_response(fund_name, f"LLM error: {e}", "no_context")

    # ── Step 4: Format citation footer ──────────────────────────────────────
    updated_display = last_updated[:10] if last_updated else "Unknown"
    citation = f"\n\n📄 Source: {source_url} | Last Updated: {updated_display}"

    return {
        "answer":       answer + citation,
        "source_url":   source_url,
        "last_updated": last_updated,
        "fund_name":    fund_name,
        "refused":      False,
        "refusal_type": "",
    }


def _refusal_response(fund_name: str, message: str, refusal_type: str) -> dict:
    return {
        "answer":       message,
        "source_url":   "",
        "last_updated": "",
        "fund_name":    fund_name,
        "refused":      True,
        "refusal_type": refusal_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test runner
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # (fund_id, question, expected_type)
    ("hdfc_large_cap",  "What is the expense ratio of HDFC Large Cap Fund?",   "factual"),
    ("hdfc_elss",       "What is the lock-in period for HDFC ELSS Tax Saver?",  "factual"),
    ("hdfc_flexi_cap",  "What is the minimum SIP amount for HDFC Flexi Cap?",   "factual"),
    ("hdfc_large_cap",  "What is the benchmark index for HDFC Large Cap Fund?", "factual"),
    ("hdfc_elss",       "Should I invest in HDFC ELSS Tax Saver?",              "advice_refusal"),
    ("hdfc_flexi_cap",  "Can you recommend the best mutual fund for me?",       "advice_refusal"),
    ("hdfc_large_cap",  "My PAN is ABCDE1234F, what fund should I buy?",        "pii_refusal"),
    ("hdfc_large_cap",  "What is the capital of France?",                       "no_context"),
]

if __name__ == "__main__":
    print("=" * 70)
    print("HDFC MF RAG Backend — Test Runner")
    print("=" * 70)

    passed = 0
    failed = 0

    for fund_id, question, expected_type in TEST_CASES:
        print(f"\n{'─'*70}")
        print(f"Fund   : {FUND_DISPLAY_NAMES.get(fund_id, fund_id)}")
        print(f"Query  : {question}")
        print(f"Expect : {expected_type}")
        print()

        result = query_rag(fund_id=fund_id, question=question)

        print(f"Answer :\n{result['answer']}")

        if expected_type == "factual":
            ok = not result["refused"]
        elif expected_type == "advice_refusal":
            ok = result["refused"] and result["refusal_type"] == "advice"
        elif expected_type == "pii_refusal":
            ok = result["refused"] and result["refusal_type"] == "pii"
        elif expected_type == "no_context":
            ok = result["refused"] and result["refusal_type"] == "no_context"
        else:
            ok = True

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n[{status}] refused={result['refused']} | type={result['refusal_type']}")

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests.")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
