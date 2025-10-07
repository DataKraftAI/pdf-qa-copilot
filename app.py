import os, io, hashlib, re
from typing import List, Tuple
import numpy as np
import streamlit as st

# Prefer PyMuPDF (cleaner text), fall back to pypdf
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False

from openai import OpenAI

# ------------------------- Page & small CSS tweaks -------------------------
st.set_page_config(page_title="📄 Policy & PDF Q&A", layout="wide")
st.markdown(
    """
    <style>
      /* Hide tiny "Press Enter…" hints under inputs */
      .stTextArea small, .stTextInput small { display: none !important; }
      textarea + div small { display: none !important; }

      /* Make the top-right language picker compact */
      .lang-compact > div[data-baseweb="select"] { max-width: 230px; }
      .stButton>button[kind="primary"]{
          background-color:#ff4b4b;
          color:#fff; font-weight:700;
          padding:10px 16px; border-radius:10px; border:none;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Header row with true right-aligned select ----------
hdr_left, hdr_spacer, hdr_right = st.columns([0.62, 0.18, 0.20], gap="small")

with hdr_left:
    st.title("📄 Policy & PDF Q&A")
    st.caption("Upload PDFs (policy, contract, handbook), then ask natural-language questions. Answers cite pages.")

with hdr_right:
    st.markdown("**Language**", help="Sprache")
    # Collapse the widget label so it stays compact; column keeps it at the right
    lang = st.selectbox(
        "",
        ["English", "Deutsch"],
        index=0,
        key="hdr_lang",
        label_visibility="collapsed",
        help="Answer language follows this selection.",
    )
    st.markdown('<div class="lang-compact"></div>', unsafe_allow_html=True)

TXT = {
    "English": {
        "settings": "Settings",
        "upload": "Upload one or more PDFs",
        "ask": "Ask a question",
        "placeholder": "e.g., What does the lease say about rent and the security deposit?",
        "strict": "Strict mode (answer only from the PDFs; say 'Not found' if unclear)",
        "button": "🔎 Get answer",
        "answer": "### Answer",
        "sources": "Sources used",
        "info": "⬆️ Upload PDFs to begin.",
        "status_prep": "Preparing documents…",
        "status_ready": "Index ready.",
        "status_ans": "Answering…",
        "no_text": "No extractable text.",
        "no_key": "No API key found.",
        "idx_not_ready": "Index not ready.",
        "budget_reached": "⚠️ Demo budget reached."
    },
    "Deutsch": {
        "settings": "Einstellungen",
        "upload": "Eine oder mehrere PDFs hochladen",
        "ask": "Frage stellen",
        "placeholder": "z. B. Was sagt der Mietvertrag über Miete und Kaution?",
        "strict": "Strikter Modus (nur aus PDFs antworten; falls unklar: 'Nicht gefunden')",
        "button": "🔎 Antwort erhalten",
        "answer": "### Antwort",
        "sources": "Verwendete Quellen",
        "info": "⬆️ Laden Sie PDFs hoch, um zu beginnen.",
        "status_prep": "Dokumente werden vorbereitet…",
        "status_ready": "Index bereit.",
        "status_ans": "Antwort wird erstellt…",
        "no_text": "Kein extrahierbarer Text.",
        "no_key": "Kein API-Schlüssel gefunden.",
        "idx_not_ready": "Index ist noch nicht bereit.",
        "budget_reached": "⚠️ Demo-Budget erreicht."
    }
}[lang]

# ------------------------- Sidebar (budget + advanced) -------------------------
with st.sidebar:
    st.header(TXT["settings"])
    BUDGET_DOLLARS = float(st.secrets.get("BUDGET_DOLLARS", os.getenv("BUDGET_DOLLARS", "5")))
    st.write(f"Demo budget: **${BUDGET_DOLLARS:.2f}/month**")

    PRICE_IN_PER_M   = float(st.secrets.get("PRICE_IN_PER_M",   "0.15"))
    PRICE_OUT_PER_M  = float(st.secrets.get("PRICE_OUT_PER_M",  "0.60"))
    PRICE_EMB_PER_M  = float(st.secrets.get("PRICE_EMB_PER_M",  "0.02"))

    with st.expander("Advanced"):
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2)
        TOP_K = st.slider("Context chunks (K)", 2, 8, 4)
        CHARS_PER_CHUNK = st.slider("Chunk size (chars)", 1000, 4000, 2000, step=250)
        polish_output = st.checkbox("Polish final answer", value=True)

# ------------------------- Upload -------------------------
uploaded = st.file_uploader(TXT["upload"], type=["pdf"], accept_multiple_files=True)

# ------------------------- Helpers -------------------------
NBSP = "\u00A0"
SOFT_HY = "\u00AD"
LIGATURES = {"\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl"}

def normalize_pdf_text(txt: str) -> str:
    if not txt: return ""
    for k,v in LIGATURES.items(): txt = txt.replace(k,v)
    txt = txt.replace(NBSP," ").replace(SOFT_HY,"")
    txt = txt.replace("**","").replace("*","").replace("_"," ")
    txt = re.sub(r"[ \t\r\f\v]+"," ",txt)
    txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)
    txt = re.sub(r"([A-Za-z])(\d)", r"\1 \2", txt)
    txt = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", txt)
    txt = re.sub(r"\s*,\s*", ", ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def read_pdf_text_pymupdf(file_bytes: bytes) -> List[Tuple[int,str]]:
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            words = page.get_text("words") or []
            words.sort(key=lambda w: (round(w[1],1), w[0]))
            lines, current_line = [], []
            current_y, last_x1 = None, None
            GAP = 2.0
            for x0,y0,x1,y1,wtext,*_ in words:
                if current_y is None or abs(y0-current_y) > 2.0:
                    if current_line: lines.append("".join(current_line)); current_line=[]
                    current_y=y0; last_x1=None
                if last_x1 is not None and (x0-last_x1) > GAP:
                    current_line.append(" ")
                current_line.append(wtext); last_x1=x1
            if current_line: lines.append("".join(current_line))
            raw = "\n".join(lines)
            pages.append((i, normalize_pdf_text(raw)))
    return pages

def read_pdf_text_pypdf(file_bytes: bytes) -> List[Tuple[int,str]]:
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages, start=1):
        try: txt = page.extract_text() or ""
        except Exception: txt = ""
        pages.append((i, normalize_pdf_text(txt)))
    return pages

def read_pdf_pages(file) -> List[Tuple[int,str]]:
    file_bytes = file.read()
    if HAVE_PYMUPDF:
        try: return read_pdf_text_pymupdf(file_bytes)
        except Exception: pass
    if HAVE_PYPDF:
        try: return read_pdf_text_pypdf(file_bytes)
        except Exception: pass
    return []

def chunk_pages(pages: List[Tuple[int,str]], chars_per_chunk=2000, overlap=200):
    chunks = []
    buf, start_page, end_page = "", None, None
    for pno, text in pages:
        if not text: continue
        pos=0
        while pos < len(text):
            take = text[pos:pos+chars_per_chunk]
            if not buf: start_page = pno
            buf += ("" if not buf else "\n") + take
            end_page = pno
            if len(buf) >= chars_per_chunk:
                chunks.append(((start_page, end_page), buf))
                buf = buf[-overlap:]; start_page = pno
            pos += chars_per_chunk
    if buf: chunks.append(((start_page, end_page), buf))
    return chunks

def approx_tokens(s: str) -> int: return max(1, int(len(s)/4))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def hash_docs(files) -> str:
    h = hashlib.sha256()
    for f in files:
        f.seek(0); h.update(f.read()); f.seek(0)
    return h.hexdigest()[:16]

def clean_answer(txt: str) -> str:
    if not txt: return txt
    txt = txt.replace("\r","")
    txt = re.sub(r"[ \t\f\v]+"," ",txt)
    txt = re.sub(r"\n[ \t]+","\n",txt)
    txt = re.sub(r"\n{3,}","\n\n",txt)
    txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)
    txt = re.sub(r"([A-Za-z])(\d)", r"\1 \2", txt)
    txt = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", txt)
    txt = re.sub(r"\s*,\s*", ", ", txt)
    return txt.strip()

# ------------------------- OpenAI client -------------------------
def get_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not key:
        st.error(TXT["no_key"]); st.stop()
    return OpenAI(api_key=key)

# ------------------------- Budget -------------------------
if "spent_cents" not in st.session_state: st.session_state.spent_cents = 0.0
def check_and_add_cost(input_toks=0, output_toks=0, embed_toks=0):
    cost = (input_toks/1e6)*PRICE_IN_PER_M + (output_toks/1e6)*PRICE_OUT_PER_M + (embed_toks/1e6)*PRICE_EMB_PER_M
    st.session_state.spent_cents += cost * 100
    if (st.session_state.spent_cents/100) > BUDGET_DOLLARS:
        st.error(TXT["budget_reached"]); st.stop()

# ------------------------- Build / load index -------------------------
index_ready = False
if uploaded:
    cache_key = hash_docs(uploaded)
    if "index_cache" not in st.session_state: st.session_state.index_cache = {}
    if cache_key in st.session_state.index_cache:
        chunks, metas, emb_matrix = st.session_state.index_cache[cache_key]
        index_ready = True
    else:
        status = st.status(TXT["status_prep"], expanded=True)
        all_chunks, metas = [], []
        for uf in uploaded:
            pages = read_pdf_pages(uf)
            chs = chunk_pages(pages, chars_per_chunk=CHARS_PER_CHUNK)
            for (p1, p2), text in chs:
                all_chunks.append(text); metas.append({"file": uf.name, "pages": (p1, p2)})
        if not all_chunks:
            status.update(label=TXT["no_text"], state="error"); st.stop()

        client = get_client()
        embed_input_tokens = sum(approx_tokens(c) for c in all_chunks)
        check_and_add_cost(embed_toks=embed_input_tokens)
        emb_resp = client.embeddings.create(model="text-embedding-3-small", input=all_chunks)
        emb_matrix = np.array([e.embedding for e in emb_resp.data], dtype=np.float32)
        st.session_state.index_cache[cache_key] = (all_chunks, metas, emb_matrix)
        chunks, metas = all_chunks, metas
        index_ready = True
        status.update(label=TXT["status_ready"], state="complete", expanded=False)

# ------------------------- QA UI -------------------------
if uploaded:
    st.subheader(TXT["ask"])
    q = st.text_area(" ", placeholder=TXT["placeholder"], height=80)
    strict = st.checkbox(TXT["strict"], value=True)

    if st.button(TXT["button"], type="primary"):
        if not index_ready:
            st.warning(TXT["idx_not_ready"]); st.stop()

        client = get_client()
        status = st.status(TXT["status_ans"], expanded=True)

        # Retrieve
        q_tokens = approx_tokens(q); check_and_add_cost(embed_toks=q_tokens)
        q_emb = client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding
        q_vec = np.array(q_emb, dtype=np.float32)
        matrix = st.session_state.index_cache[hash_docs(uploaded)][2]
        sims = [cosine_sim(q_vec, emb) for emb in matrix]
        top_idx = np.argsort(sims)[::-1][:TOP_K]

        selected = []
        cache = st.session_state.index_cache[hash_docs(uploaded)]
        for i in top_idx:
            text = cache[0][int(i)]; meta = cache[1][int(i)]
            p1, p2 = meta["pages"]
            source = f'{meta["file"]} (pages {p1}-{p2})' if p1 != p2 else f'{meta["file"]} (page {p1})'
            selected.append((source, text))

        context_blocks = [f"[Source: {src}]\n{txt}" for src, txt in selected]
        context = "\n\n---\n\n".join(context_blocks)

        # Make the model answer in the selected language
        answer_lang = "German" if lang == "Deutsch" else "English"
        language_clause = f"Please answer in {answer_lang}."

        # Guardrails / formatting
        guardrails = "Answer using only provided sources. If unclear, say 'Not found'." if lang=="English" else \
                     "Antworte ausschließlich mit den bereitgestellten Quellen. Wenn unklar, sage 'Nicht gefunden'."
        format_enforce = "Format clean Markdown, use bullets if needed."
        formatting_hint = "Return amounts as short bullets if asked (Rent, Deposit, Fees)."

        user_prompt = f"""
{language_clause}

Question:
{q}

{guardrails}

{format_enforce}

{formatting_hint}

Sources:
{context}
""".strip()

        in_tokens = approx_tokens(user_prompt); out_tokens = 500
        check_and_add_cost(input_toks=in_tokens, output_toks=out_tokens)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=out_tokens,
        )
        raw_answer = resp.choices[0].message.content
        answer = clean_answer(raw_answer)

        st.markdown(TXT["answer"])
        st.markdown(answer)
        with st.expander(TXT["sources"]):
            for src, _ in selected:
                st.write(f"• {src}")
else:
    st.info(TXT["info"])
