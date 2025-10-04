import streamlit as st
import openai
import fitz  # PyMuPDF
import re

# --- Page setup ---
st.set_page_config(page_title="Policy / PDF Q&A", page_icon="📄", layout="centered")

# Hide the hint under the text field
st.markdown(
    """
    <style>
      /* Hide the tiny helper hint under the text area */
      .stTextArea small { display: none !important; }
      /* Extra safety: hide any small hint right after a textarea */
      textarea + div small { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Policy & PDF Q&A")

# --- File upload ---
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# --- Extract text from PDFs ---
def extract_text_from_pdfs(files):
    texts = []
    for f in files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                texts.append(f"[page {page_num}] {text}")
    return "\n".join(texts)

# --- Clean answer formatting ---
def clean_answer(txt: str) -> str:
    if not txt:
        return txt

    # Remove weird linebreak artifacts
    txt = txt.replace("\r", "")

    # Collapse spaces/tabs but NOT newlines
    txt = re.sub(r"[ \t\f\v]+", " ", txt)
    # Clean spaces after newlines
    txt = re.sub(r"\n[ \t]+", "\n", txt)
    # Collapse too many blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    # Spacing between numbers and letters
    txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)
    txt = re.sub(r"([A-Za-z])(\d)", r"\1 \2", txt)

    # Fix number grouping (1,000 not 1, 000)
    txt = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", txt)

    # Normalize commas
    txt = re.sub(r"\s*,\s*", ", ", txt)

    # Tidy
    txt = txt.strip()

    # Common replacements
    fixes = {
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        " $ ": " $",
        "€ ": "€",
    }
    for k, v in fixes.items():
        txt = txt.replace(k, v)

    return txt

# --- Get answer from OpenAI ---
def ask_gpt(question, context, strict_mode=True):
    system_prompt = (
        "You are a precise assistant. Answer ONLY from the given PDF text.\n"
        "Rules:\n"
        "- If answer is unclear, say 'Not found in the documents.'\n"
        "- Always cite the page like [page 2].\n"
        "- Present the answer in clean Markdown.\n"
        "- Use proper bullet points with one item per line when listing.\n"
        "- Fix spacing/formatting artifacts.\n"
        "- Do not invent content.\n"
    )
    if not strict_mode:
        system_prompt += "- If relevant, you may reason using general knowledge too.\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=500,
    )

    answer = resp["choices"][0]["message"]["content"]
    return clean_answer(answer)

# --- UI ---
if uploaded_files:
    context_text = extract_text_from_pdfs(uploaded_files)
    question = st.text_area("Type your question", height=100)
    strict_mode = st.checkbox("Strict mode (answer only from the PDFs; say 'Not found' if unclear)", value=True)

    if st.button("🔎 Get answer"):
        with st.spinner("Thinking..."):
            answer_text = ask_gpt(question, context_text, strict_mode)
        st.success("Done.")
        st.markdown("**Answer**")
        st.markdown(answer_text)
        with st.expander("Sources used"):
            st.text(context_text[:2000] + "...\n\n[truncated]")
