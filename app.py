import os, io, math, hashlib, re
from typing import List, Tuple
import numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ---------- Page ----------
st.set_page_config(page_title="📄 Policy & PDF Q&A", layout="wide")
st.title("📄 Policy & PDF Q&A")
st.caption("Upload PDFs (policy, contract, handbook), then ask natural-language questions. Answers cite pages.")

# ---------- Sidebar (budget + model) ----------
with st.sidebar:
    st.header("Settings")
    # Hard budget (demo guard). Also set a hard limit in your OpenAI dashboard.
    BUDGET_DOLLARS = float(st.secrets.get("BUDGET_DOLLARS", os.getenv("BUDGET_DOLLARS", "5")))
    st.write(f"Demo budget: **${BUDGET_DOLLARS:.2f}/month** (app guard)")

    # Prices per 1M tokens (USD) – gpt-4o-mini + embeddings small
    PRICE_IN_PER_M   = float(st.secrets.get("PRICE_IN_PER_M",   "0.15"))  # input
    PRICE_OUT_PER_M  = float(st.secrets.get("PRICE_OUT_PER_M",  "0.60"))  # output
    PRICE_EMB_PER_M  = float(st.secrets.get("PRICE_EMB_PER_M",  "0.02"))  # embeddings

    # Advanced (can hide later)
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
                # keep small overlap
                buf = buf[-overlap:]
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
    Light cleanup for PDF extraction artifacts the model may echo.
    - collapse excessive whitespace
    - fix missing spaces between numbers/words: '1000and' -> '1000 and'
    - normalize spaces around commas
    """
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)     # 1000and -> 1000 and
    txt = re.sub(r"\s*,\s*", ", ", txt)                # , spacing
    return txt.strip()

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
    # Convert to USD cents using current prices
    cost = (input_toks/1e6)*PRICE_IN_PER_M + (output_toks/1e6)*PRICE_OUT_PER_M + (embed_toks/1e6)*PRICE_EMB_PER_M
    st.session_state.spent_cents += cost * 100
    if (st.session_state.spent_cents/100) > BUDGET_DOLLARS:
        st.error("⚠️ Demo budget reached. Please try again next month.")
        st.stop()

# ---------- Build index when files uploaded ----------
index_ready = False
if uploaded:
    # cache by content hash within the session
    cache_key = hash_docs(uploaded)
    if "index_cache" not in st.session_state:
        st.session_state.index_cache = {}
    if cache_key in st.session_state.index_cache:
        chunks, metas, emb_matrix = st.session_state.index_cache[cache_key]
        index_ready = True
    else:
        st.info("Building index (embedding chunks). This runs only once per upload…")
        all_chunks, metas = [], []
        for uf in uploaded:
            pages = read_pdf_text(uf)
            # note: scanned PDFs (images) will produce little or no text
            chs = chunk_pages(pages, chars_per_chunk=CHARS_PER_CHUNK)
            for (p1, p2), text in chs:
                all_chunks.append(text)
                metas.append({"file": uf.name, "pages": (p1, p2)})

        if not all_chunks:
            st.error("No extractable text found. (Scanned PDFs are not supported without OCR.)")
            st.stop()

        # embed chunks
        client = get_client()
        embed_input_tokens = sum(approx_tokens(c) for c in all_chunks)
        check_and_add_cost(embed_toks=embed_input_tokens)

        with st.spinner("Embedding document chunks…"):
            emb_resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=all_chunks
            )
        emb_matrix = np.array([e.embedding for e in emb_resp.data], dtype=np.float32)

        st.session_state.index_cache[cache_key] = (all_chunks, metas, emb_matrix)
        chunks, metas = all_chunks, metas
        index_ready = True

# ---------- QA UI ----------
if uploaded:
    st.subheader("Ask a question")
    q = st.text_input("Example: What does the leave policy say about sick days?")
    strict = st.checkbox("Strict mode (answer only from the PDFs; say 'Not found' if unclear)", value=True)

    if st.button("🔎 Get answer"):
        if not index_ready:
            st.warning("Index not ready yet. Try again.")
            st.stop()

        client = get_client()

        # Retrieve top-K chunks by cosine sim
        # Embed the query
        q_tokens = approx_tokens(q)
        check_and_add_cost(embed_toks=q_tokens)
        with st.spinner("Thinking…"):
            q_emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=q
            ).data[0].embedding

        q_vec = np.array(q_emb, dtype=np.float32)
        sims = [cosine_sim(q_vec, emb) for emb in st.session_state.index_cache[hash_docs(uploaded)][2]]
        top_idx = np.argsort(sims)[::-1][:TOP_K]

        # Build context
        selected = []
        for i in top_idx:
            text = st.session_state.index_cache[hash_docs(uploaded)][0][int(i)]
            meta = st.session_state.index_cache[hash_docs(uploaded)][1][int(i)]
            p1, p2 = meta["pages"]
            source = f'{meta["file"]} (pp. {p1}-{p2})' if p1 != p2 else f'{meta["file"]} (p. {p1})'
            selected.append((source, text))

        context_blocks = []
        for src, txt in selected:
            context_blocks.append(f"[Source: {src}]\n{txt}")
        context = "\n\n---\n\n".join(context_blocks)

        # Guardrails text
        if strict:
            guardrails = (
                "Answer using only the provided sources.\n"
                "Quote briefly and **cite the page(s)** like [p. 3] or [Policy.pdf p. 12].\n"
                "Present the answer in **clean, human-readable sentences**. "
                "Fix spacing/formatting artifacts from the PDF text (numbers, commas, words). "
                "Do **not** invent content. If unclear or not present, say **'Not found in the documents.'**"
            )
        else:
            guardrails = (
                "Prefer the provided sources; if unclear, you may infer cautiously.\n"
                "Quote briefly and **cite the page(s)** like [p. 3] or [Policy.pdf p. 12].\n"
                "Present the answer in **clean, human-readable sentences**. "
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

        # Rough token estimate for budget guard
        in_tokens = approx_tokens(user_prompt)
        out_tokens = 500  # cap
        check_and_add_cost(input_toks=in_tokens, output_toks=out_tokens)

        # Call LLM
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=out_tokens,
        )
        raw_answer = resp.choices[0].message.content
        answer = clean_answer(raw_answer)  # post-process

        st.markdown("### Answer")
        st.markdown(answer)

        # Show sources used
        with st.expander("Sources used"):
            for src, _txt in selected:
                st.write(f"• {src}")

else:
    st.info("⬆️ Upload PDFs to begin.")
