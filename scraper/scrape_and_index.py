"""
HDFC Mutual Fund — Data Scraper & ChromaDB Indexer
===================================================
Phase 1 of the RAG pipeline.

What this script does:
  1. Uses Playwright to navigate the HDFC Fund document pages (KIM + Scheme Summary).
  2. Searches for each of the 3 target funds using the page's search bar.
  3. Extracts the direct PDF download URL for each fund/doc-type combination.
  4. Downloads each PDF to a temp buffer and computes an MD5 checksum.
  5. Compares the checksum against metadata.json — skips if unchanged.
  6. If changed (or first run): extracts text, chunks it, embeds it, and upserts into ChromaDB.
  7. Updates metadata.json with the new hash and timestamp.

Usage:
  python scraper/scrape_and_index.py

Requirements:
  pip install -r requirements.txt
  playwright install chromium
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chromadb
import pdfplumber
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = ROOT_DIR / "metadata.json"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
PDF_CACHE_DIR = ROOT_DIR / "data" / "pdfs"

# ─── Load env ─────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(VECTOR_DB_DIR))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hdfc_mf_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Target Funds ─────────────────────────────────────────────────────────────
FUNDS = {
    "hdfc_large_cap": "HDFC Large Cap Fund",
    "hdfc_elss":      "HDFC ELSS Tax Saver",
    "hdfc_flexi_cap": "HDFC Flexi Cap Fund",
}

SOURCE_PAGES = {
    "kim":            "https://www.hdfcfund.com/mutual-funds/fund-documents/kim",
    "scheme_summary": "https://www.hdfcfund.com/mutual-funds/fund-documents/scheme-summary",
}

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict) -> None:
    tmp = METADATA_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    tmp.replace(METADATA_FILE)
    log.info("metadata.json updated.")


def get_current_md5(fund_id: str, doc_type: str, metadata: dict) -> Optional[str]:
    return metadata.get(fund_id, {}).get(doc_type, {}).get("md5")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Web scraper (Playwright)
# ═══════════════════════════════════════════════════════════════════════════════

async def find_pdf_url(page, fund_name: str, source_url: str, doc_type: str) -> Optional[str]:
    """Navigate to HDFC fund doc page, use search bar to filter, then extract PDF URL."""
    log.info(f"  -> Navigating to {source_url}")
    await page.goto(source_url, wait_until="networkidle", timeout=60_000)
    await page.wait_for_timeout(2000)

    # Try search bar first to filter the fund list
    searched = await _apply_search_bar(page, fund_name)
    if searched:
        await page.wait_for_timeout(2000)

    # Now extract the PDF link from the (possibly filtered) page
    pdf_url = await _extract_pdf_link(page, fund_name, doc_type)
    if pdf_url:
        log.info(f"  Found via search bar: {pdf_url}")
        return pdf_url

    log.info(f"  Search bar failed for '{fund_name}', falling back to pagination.")
    return await _pagination_strategy(page, fund_name, source_url, doc_type)


async def _apply_search_bar(page, fund_name: str) -> bool:
    """Fill the search bar and return True if found."""
    try:
        search_selectors = [
            "input[placeholder*='Search']",
            "input[placeholder*='search']",
            "input[type='search']",
            "input[type='text']",
        ]
        for sel in search_selectors:
            if await page.locator(sel).count() > 0:
                search_input = page.locator(sel).first
                await search_input.click()
                await search_input.fill(fund_name)
                return True
    except Exception as e:
        log.debug(f"  Search bar error: {e}")
    return False


async def _extract_pdf_link(page, fund_name: str, doc_type: str) -> Optional[str]:
    """
    Find the PDF download link for the given fund on the current page.

    Strategy is doc_type-aware:
    - KIM: PDF URLs encode the fund name in the path (e.g. 'KIM%20-%20HDFC%20Large%20Cap...')
           -> match by fund keywords appearing in the href
    - scheme_summary: PDF URLs are opaque codes (e.g. HDFCT2.pdf)
           -> use JS to find any element whose visible text contains the fund name,
              walk up to its card container, and return the .pdf sibling link
    Both doc types fall back to a combined scan if primary fails.
    """
    fund_name_lower = fund_name.lower()
    fund_words = fund_name.split()
    # Distinctive words (skip "HDFC", require length > 3)
    distinctive_words = [w.lower() for w in fund_words if w.lower() != "hdfc" and len(w) > 3]

    # ── KIM primary: match fund keywords in href ───────────────────────────────
    if doc_type == "kim":
        try:
            all_links = await page.eval_on_selector_all(
                "a",
                "elements => elements.map(el => ({ href: el.href || '' }))"
            )
            for link in all_links:
                href = link.get("href", "")
                if not href or ".pdf" not in href.lower():
                    continue
                if "hdfcfund.com" not in href:
                    continue
                href_decoded = href.lower().replace("%20", " ")
                # Require at least 2 of the distinctive words in the href
                matches = sum(1 for w in distinctive_words if w in href_decoded)
                if matches >= min(2, len(distinctive_words)):
                    log.debug(f"    [KIM-href] {href[:80]}")
                    return href
        except Exception as e:
            log.debug(f"    KIM href strategy error: {e}")

    # ── Scheme Summary primary: JS card-walk by visible text ──────────────────
    if doc_type == "scheme_summary":
        try:
            result = await page.evaluate(
                """([fundNameLower]) => {
                    const snippet = fundNameLower.substring(0, 14);
                    const allEls = Array.from(document.querySelectorAll('p, div, span, h3, h4, li'));
                    for (const el of allEls) {
                        const txt = (el.innerText || el.textContent || '').toLowerCase();
                        // Only match elements where this text is the primary content
                        // (not ones that contain everything on the page)
                        if (txt.includes(snippet) && txt.length < 120) {
                            let container = el;
                            for (let i = 0; i < 6; i++) {
                                container = container.parentElement;
                                if (!container) break;
                                const pdfLinks = container.querySelectorAll('a[href]');
                                for (const a of pdfLinks) {
                                    const h = a.href || '';
                                    if (h.includes('hdfcfund.com') && h.includes('.pdf')) {
                                        return h;
                                    }
                                }
                            }
                        }
                    }
                    return null;
                }""",
                [fund_name_lower]
            )
            if result:
                log.debug(f"    [SS-cardwalk] {result[:80]}")
                return result
        except Exception as e:
            log.debug(f"    Scheme summary card-walk error: {e}")

    # ── Fallback: combined href + card text scan ───────────────────────────────
    try:
        all_links = await page.eval_on_selector_all(
            "a",
            """elements => elements.map(el => ({
                href: el.href || '',
                text: ((el.closest('[class]') || el.parentElement || el).innerText || '')
            }))"""
        )
        for link in all_links:
            href = link.get("href", "")
            card_text = link.get("text", "").lower()
            if not href or ".pdf" not in href.lower():
                continue
            if "hdfcfund.com" not in href:
                continue
            href_decoded = href.lower().replace("%20", " ")
            name_in_href = sum(1 for w in distinctive_words if w in href_decoded) >= 1
            name_in_card = fund_name_lower[:12] in card_text
            if name_in_href or name_in_card:
                log.debug(f"    [fallback] {href[:80]}")
                return href
    except Exception as e:
        log.debug(f"    Fallback strategy error: {e}")

    return None


async def _pagination_strategy(page, fund_name: str, source_url: str, doc_type: str) -> Optional[str]:
    """Iterate through pagination pages until fund found."""
    page_num = 1
    while True:
        log.info(f"  -> Checking page {page_num}...")
        url = await _extract_pdf_link(page, fund_name, doc_type)
        if url:
            log.info(f"  Found on page {page_num}: {url}")
            return url
        next_clicked = await _click_next_page(page, page_num)
        if not next_clicked:
            log.warning(f"  '{fund_name}' not found after page {page_num}.")
            return None
        page_num += 1
        await page.wait_for_timeout(1500)


async def _click_next_page(page, current_page: int) -> bool:
    """Attempt to click the next pagination button."""
    next_page_num = current_page + 1
    selectors_to_try = [
        f"li.page-item a.page-link:has-text('{next_page_num}')",
        f"button:has-text('{next_page_num}')",
        f"a:has-text('{next_page_num}')",
        "li.page-item.active + li a.page-link",
        "a[aria-label='Next']",
        "button[aria-label='Next']",
        ".pagination .next a",
        "li:has-text('Next') a",
    ]
    for sel in selectors_to_try:
        try:
            locator = page.locator(sel).first
            if await locator.count() > 0:
                await locator.click()
                await page.wait_for_timeout(1500)
                return True
        except Exception:
            continue
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PDF processing
# ═══════════════════════════════════════════════════════════════════════════════

def download_pdf(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.hdfcfund.com/",
    }
    log.info(f"    Downloading PDF: {url[:80]}...")
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.content


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def extract_text_from_pdf(data: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_parts.append(f"[Page {page_num}]\n{text.strip()}")
    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ChromaDB operations
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection():
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def delete_existing_vectors(collection, fund_id: str, doc_type: str) -> None:
    where_filter = {"$and": [{"fund_id": fund_id}, {"doc_type": doc_type}]}
    results = collection.get(where=where_filter, include=[])
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        log.info(f"    Deleted {len(ids_to_delete)} old vectors for {fund_id}/{doc_type}.")
    else:
        log.info(f"    No existing vectors to delete for {fund_id}/{doc_type}.")


def upsert_chunks(
    collection,
    chunks: List[str],
    fund_id: str,
    doc_type: str,
    source_url: str,
    last_updated: str,
) -> None:
    documents = []
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{fund_id}__{doc_type}__{i:04d}"
        documents.append(chunk)
        metadatas.append({
            "fund_id":      fund_id,
            "doc_type":     doc_type,
            "source_url":   source_url,
            "last_updated": last_updated,
            "chunk_index":  i,
        })
        ids.append(chunk_id)
    batch_size = 100
    for batch_start in range(0, len(documents), batch_size):
        batch_end = batch_start + batch_size
        collection.add(
            documents=documents[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
            ids=ids[batch_start:batch_end],
        )
    log.info(f"    Inserted {len(documents)} chunks into ChromaDB.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

async def process_fund_doc(
    page,
    fund_id: str,
    fund_name: str,
    doc_type: str,
    source_page_url: str,
    metadata: dict,
    collection,
) -> None:
    log.info(f"\n{'─'*60}")
    log.info(f"Processing: {fund_name} | {doc_type.upper()}")
    log.info(f"{'─'*60}")

    pdf_url = await find_pdf_url(page, fund_name, source_page_url, doc_type)
    if not pdf_url:
        log.error(f"  Could not find PDF URL for '{fund_name}' ({doc_type}). Skipping.")
        return

    log.info(f"  PDF URL: {pdf_url}")

    try:
        pdf_bytes = download_pdf(pdf_url)
    except Exception as e:
        log.error(f"  Download failed for {pdf_url}: {e}")
        return

    new_md5 = compute_md5(pdf_bytes)
    stored_md5 = get_current_md5(fund_id, doc_type, metadata)

    if new_md5 == stored_md5:
        log.info(f"  No change detected for '{fund_name}' ({doc_type}). Hash: {new_md5[:8]}...")
        return

    log.info(f"  Change detected! Old MD5: {stored_md5 or 'none'} -> New: {new_md5[:8]}...")

    log.info("    Extracting text from PDF...")
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        log.error(f"    Text extraction failed: {e}")
        return

    if not text.strip():
        log.warning("    Extracted text is empty — PDF may be image-only. Skipping.")
        return

    log.info(f"    Extracted {len(text):,} characters.")
    chunks = chunk_text(text)
    log.info(f"    Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        delete_existing_vectors(collection, fund_id, doc_type)
        upsert_chunks(collection, chunks, fund_id, doc_type, pdf_url, timestamp)
    except Exception as e:
        log.error(f"    ChromaDB upsert failed: {e}")
        return

    if fund_id not in metadata:
        metadata[fund_id] = {}
    metadata[fund_id][doc_type] = {
        "url":          pdf_url,
        "md5":          new_md5,
        "last_updated": timestamp,
    }
    save_metadata(metadata)
    log.info(f"  Done: {fund_name} ({doc_type}) — {len(chunks)} chunks indexed.")


async def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("HDFC MF RAG — Data Pipeline Starting")
    log.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    log.info(f"Loaded metadata.json ({len(metadata)} fund entries).")

    collection = get_chroma_collection()
    log.info(f"ChromaDB collection ready: {CHROMA_COLLECTION_NAME}")
    log.info(f"Existing vector count: {collection.count()}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        for doc_type, source_url in SOURCE_PAGES.items():
            for fund_id, fund_name in FUNDS.items():
                try:
                    await process_fund_doc(
                        page            = page,
                        fund_id         = fund_id,
                        fund_name       = fund_name,
                        doc_type        = doc_type,
                        source_page_url = source_url,
                        metadata        = metadata,
                        collection      = collection,
                    )
                except Exception as e:
                    log.error(f"Unhandled error for {fund_name} ({doc_type}): {e}", exc_info=True)
                finally:
                    await asyncio.sleep(2)

        await browser.close()

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete!")
    log.info(f"Total vectors in ChromaDB: {collection.count()}")
    log.info("=" * 60)

    print("\nMetadata Summary:")
    for fid, docs in metadata.items():
        print(f"\n  {fid}:")
        for dtype, info in docs.items():
            updated = info.get("last_updated", "never")
            md5_short = (info.get("md5") or "none")[:8]
            print(f"    {dtype}: md5={md5_short}... | updated={updated}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
