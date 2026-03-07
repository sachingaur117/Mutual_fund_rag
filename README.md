# HDFC Mutual Fund RAG FAQ Assistant

A RAG-based FAQ bot for HDFC Mutual Fund factual queries.  
Powered by **ChromaDB** (vector store) + **Gemini 1.5 Flash** (LLM) + **Streamlit** (UI).

## Supported Funds
- HDFC Large Cap Fund
- HDFC ELSS Tax Saver
- HDFC Flexi Cap Fund

---

## Project Structure
```
Nextleap_MF_RAG_Agent/
├── scraper/
│   └── scrape_and_index.py   # Phase 1 — Data pipeline
├── backend/
│   └── rag_backend.py        # Phase 3 — Query engine
├── frontend/
│   └── app.py                # Phase 4 — Streamlit UI
├── vector_db/                # ChromaDB persisted store (auto-created)
├── data/pdfs/                # PDF cache dir (auto-created)
├── metadata.json             # Hash + timestamp registry per fund
├── .env                      # Secrets (copy from .env.example)
├── requirements.txt
└── README.md
```

---

## Phase 1 — Run the Scraper

### Setup
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### Configure
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (needed for Phase 3)
```

### Run
```bash
python scraper/scrape_and_index.py
```

**What happens:**
1. Navigates HDFC fund document pages (KIM + Scheme Summary)
2. Finds the 3 target funds (using search bar + pagination fallback)
3. Downloads each PDF → computes MD5 checksum
4. Skips if hash unchanged; otherwise chunks + embeds + indexes into `vector_db/`
5. Updates `metadata.json` with hash and timestamp

**Re-running** is safe — unchanged PDFs are skipped with "No change detected" log.

---

## GitHub Action (Phase 5)
Runs automatically on the 1st of every month.  
Requires `GEMINI_API_KEY` set as a GitHub Repository Secret.

---

## Key Constraints
- ≤ 3 sentence answers
- No investment advice, no PII storage
- Every answer cites the exact source PDF URL
- Public HDFC sources only
