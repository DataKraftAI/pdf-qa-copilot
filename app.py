import os, io, hashlib, re
from typing import List, Tuple
import numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ---------- Page ----------
st.set_page_config(page_title="📄 Policy & PDF Q&A", layout="wide")
st.title("📄 Policy & PDF Q&A")
st.caption("Upload PDFs (policy, contract, handbook), then ask natural-language questions. Answers cite pages.")

# ---------- Sidebar (budget + advanced) ----------
with st.sidebar:
    st.header("Settings")
    # Demo budget guard (still set a hard limit in your OpenAI dashboard)
    BUDGET_DOLLARS = float(st.secrets.get("BUDGET_DOLLARS", os.getenv("BUDGET_DOLLARS", "5")))
    st.write(f"Demo budget: **${BUDGET_DOLLARS:.2f}/month**")

    # Prices per 1M tokens (USD)
    PRICE_IN_PER_M   = float(st.secrets.get("PRICE_IN_PER_M",   "0.15"))  # gpt-4o-mini input
    PRICE_OUT_PER_M  = float(st.secrets.get("PRICE_OUT_PER_M",  "0.60"))  # gpt-4o-mini output
    PRICE_EMB_PER_M  = float(st.secrets.get("PRICE_EMB_PER_M",  "0.02"))  # text-embedding-3-small

    with st.expander("Advanced"):
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2,
                                help="Lower = strict & factual. Higher = more flexible wording.")
        TOP_K = st.slider("Context chunks (K)", 2, 8, 4)
        CHARS_PER_CHUNK = st.slider("Chunk size (chars)", 1000, 4000, 2000, step=250)

# ---------- Upload ----------
uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# ---------- Helpers ----------
def read_pdf_text(file) -> List[Tuple[int, str]]:
    """Return [(page_number, text)] for a PDF (1-indexed pages)."""
    reader = PdfReader(io.BytesIO(file.read()))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, txt.strip()))
    return pages

def chunk_pages(pages: List[Tuple[int, str]], chars_per_chunk=2000, overlap=200):
    """Greedy char chunking across pages; keep page refs in each chunk."""
    chunks = []
    buf, start_page, end_page = "", None, None
    for pno, text in pages:
        if not text:
            continue
        pos = 0
        while pos < len(text):
            take = text[pos:pos+chars_per_chunk]
            if buf == "":
                start_page = pno
            buf += ("\n" + take) if buf else take
            end_page = pno
            if len(buf) >= chars_per_chunk:
                chunks.append(((start_page, end_page), buf))
                buf = buf[-overlap:]  # small overlap
                start_page = pno
            pos += chars_per_chunk
    if buf:
        chunks.append(((start_page, end_page), buf))
    return chunks

def approx_tokens(s: str) -> int:
    return max(1, int(len(s) / 4))  # rough 1 token ≈ 4 chars

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def hash_docs(files) -> str:
    h = hashlib.sha256()
    for f in files:
        f.seek(0); h.update(f.read()); f.seek(0)
    return h.hexdigest()[:16]

def clean_answer(txt: str) -> str:
    """
    Strong cleanup for PDF artifacts the model may echo.
    - remove stray *, ** and underscores (markdown / OCR noise)
    - collapse whitespace
    - fix missing spaces between digits/letters
    - normalize thousands: '1, 000' -> '1,000'
    - normalize spaces around commas
    - patch common glued phrases from leases/policies
    """
    # strip markdown-ish markers
    txt = txt.replace("**", "").replace("*", "").replace("_", " ")

    # whitespace & number/letter spacing
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)
    txt = re.sub(r"([A-Za-z])(\d)", r"\1 \2", txt)

    # 1, 000 -> 1,000
    txt = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", txt)

    # spaces around commas
    txt = re.sub(r"\s*,\s*", ", ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()

    fixes = {
        "permonth": "per month",
        "isalso": "is also",
        "andthe": "and the",
        "andthere": "and there",
        "securitydeposit": "security deposit",
        "refundabledeposit": "refundable deposit",
        "depositis": "deposit is",
        "rentis": "rent is",
        "dueprior": "due prior",
    }
    low = txt.lower()
    for bad, good in fixes.items():
        if bad in low:
            txt = re.sub(bad, good, txt, flags=re.IGNORECASE)

    return txt

# ---------- OpenAI client ----------
def get_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not key:
        st.error("No API key found. Add OPENAI_API_KEY in Streamlit → Settings → Secrets.")
        st.stop()
    return OpenAI(api_key=key)

# ---------- Budget tracking (demo guard) ----------
if "spent_cents" not in st.session_state:
    st.session_state.spent_cents = 0.0

def check_and_add_cost(input_toks=0, output_toks=0, embed_toks=0):
    cost = (input_toks/1e6)*PRICE_IN_PER_M + (output_toks/1e6)*PRICE_OUT_PER_M + (embed_toks/1e6)*PRICE_EMB_PER_M
    st.session_state.spent_cents += cost * 100
    if (st.session_state.spent_cents/100) > BUDGET_DOLLARS:
        st.error("⚠️ Demo budget reached. Please try again next month.")
        st.stop()

# ---------- Build index when files uploaded ----------
index_ready = False
if uploaded:
    cache_key = hash_docs(uploaded)
    if "index_cache" not in st.session_state:
        st.session_state.index_cache = {}
    if cache_key in st.session_state.index_cache:
        chunks, metas, emb_matrix = st.session_state.index_cache[cache_key]
        index_ready = True
    else:
        status = st.status("Preparing your documents…", expanded=True)
        status.update(label="Extracting pages…", state="running")

        all_chunks, metas = [], []
        for uf in uploaded:
            pages = read_pdf_text(uf)
            chs = chunk_pages(pages, chars_per_chunk=CHARS_PER_CHUNK)
            for (p1, p2), text in chs:
                all_chunks.append(text)
                metas.append({"file": uf.name, "pages": (p1, p2)})

        if not all_chunks:
            status.update(label="No extractable text found (scanned PDFs need OCR).", state="error")
            st.stop()

        client = get_client()
        embed_input_tokens = sum(approx_tokens(c) for c in all_chunks)
        check_and_add_cost(embed_toks=embed_input_tokens)

        status.update(label="Embedding chunks…", state="running")
        emb_resp = client.embeddings.create(model="text-embedding-3-small", input=all_chunks)
        emb_matrix = np.array([e.embedding for e in emb_resp.data], dtype=np.float32)

        st.session_state.index_cache[cache_key] = (all_chunks, metas, emb_matrix)
        status.update(label="Index ready.", state="complete", expanded=False)
        chunks, metas = all_chunks, metas
        index_ready = True

# ---------- QA UI ----------
if uploaded:
    st.subheader("Ask a question")
    q = st.text_area(
        "Type your question",
        placeholder="e.g., What does the lease say about rent and the security deposit?",
        height=80
    )
    strict = st.checkbox("Strict mode (answer only from the PDFs; say 'Not found' if unclear)", value=True)

    if st.button("🔎 Get answer", type="primary"):
        if not index_ready:
            st.warning("Index not ready yet. Try again.")
            st.stop()

        client = get_client()
        status = st.status("Answering…", expanded=True)

        # 1) Embed query
        status.update(label="Embedding your question…", state="running")
        q_tokens = approx_tokens(q)
        check_and_add_cost(embed_toks=q_tokens)
        q_emb = client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding

        # 2) Retrieve top-K
        status.update(label="Retrieving relevant passages…", state="running")
        q_vec = np.array(q_emb, dtype=np.float32)
        matrix = st.session_state.index_cache[hash_docs(uploaded)][2]
        sims = [cosine_sim(q_vec, emb) for emb in matrix]
        top_idx = np.argsort(sims)[::-1][:TOP_K]

        selected = []
        cache = st.session_state.index_cache[hash_docs(uploaded)]
        for i in top_idx:
            text = cache[0][int(i)]
            meta = cache[1][int(i)]
            p1, p2 = meta["pages"]
            source = f'{meta["file"]} (pages {p1}-{p2})' if p1 != p2 else f'{meta["file"]} (page {p1})'
            selected.append((source, text))

        context_blocks = [f"[Source: {src}]\n{txt}" for src, txt in selected]
        context = "\n\n---\n\n".join(context_blocks)

        # 3) Build prompt (paraphrase cleanly, cite [page N], fix artifacts)
        if strict:
            guardrails = (
                "Answer using only the provided sources.\n"
                "Paraphrase in clean, human-readable English (do not copy raw text).\n"
                "State amounts clearly with currency (e.g., $1,000) and keep units.\n"
                "Cite the page(s) like [page 3] or [Policy.pdf page 12].\n"
                "Fix spacing/formatting artifacts from the PDF text (numbers, commas, words).\n"
                "Do not invent content. If unclear or not present, say 'Not found in the documents.'"
            )
        else:
            guardrails = (
                "Prefer the provided sources; if unclear, you may infer cautiously.\n"
                "Paraphrase in clean, human-readable English (do not copy raw text).\n"
                "State amounts clearly with currency (e.g., $1,000) and keep units.\n"
                "Cite the page(s) like [page 3] or [Policy.pdf page 12].\n"
                "Fix spacing/formatting artifacts from the PDF text (numbers, commas, words)."
            )

        user_prompt = f"""
You are an expert policy analyst. Be precise and concise.

Question:
{q}

{guardrails}

Sources:
{context}
""".strip()

        # Budget guard for the chat completion
        in_tokens = approx_tokens(user_prompt)
        out_tokens = 500
        check_and_add_cost(input_toks=in_tokens, output_toks=out_tokens)

        # 4) Generate answer
        status.update(label="Generating answer…", state="running")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=out_tokens,
        )
        raw_answer = resp.choices[0].message.content
        answer = clean_answer(raw_answer)

        status.update(label="Done.", state="complete", expanded=False)

        st.markdown("### Answer")
        st.markdown(answer)

        with st.expander("Sources used"):
            for src, _txt in selected:
                st.write(f"• {src}")
else:
    st.info("⬆️ Upload PDFs to begin.")
