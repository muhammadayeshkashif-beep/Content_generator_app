# unified_generator_app.py
import os
import io
import re
import json
import tempfile
import hashlib
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch

import pdfplumber
from PyPDF2 import PdfReader

import chromadb
from chromadb.config import Settings

from openai import OpenAI  # OpenRouter-compatible client usage

# ---------------------------
# CONFIG
# ---------------------------
CHROMA_DIR = "chroma_db_unified"
BOOKS_JSON = "books_unified_meta.json"
POSTS_COLL = "posts_examples_unified"
BOOKS_COLL = "books_chunks_unified"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
GEN_MODEL = "deepseek/deepseek-r1:free"  # user-selected model

# ---------------------------
# CLIENTS
# ---------------------------
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

# Ensure collections exist
try:
    posts_collection = chroma_client.get_or_create_collection(name=POSTS_COLL)
except Exception:
    posts_collection = chroma_client.create_collection(name=POSTS_COLL)

try:
    books_collection = chroma_client.get_or_create_collection(name=BOOKS_COLL)
except Exception:
    books_collection = chroma_client.create_collection(name=BOOKS_COLL)

# ---------------------------
# EMBEDDER
# ---------------------------
class SimpleEmbedder:
    def __init__(self, model_name=EMBED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts: List[str], batch_size: int = 16):
        if isinstance(texts, str):
            texts = [texts]
        out_embs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_out = self.model(**encoded)
                emb = model_out.last_hidden_state.mean(dim=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out_embs.extend(emb.cpu().numpy().tolist())
        return out_embs

embedder = SimpleEmbedder()

# ---------------------------
# Regex patterns for headings
# ---------------------------
PART_RE = re.compile(r"^\s*(Part\s+[IVXLC0-9A-Za-z\-]+)\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*(Chapter\s+(\d+))\b", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s]{3,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|Pitfall\s*\d+|Mistake\s*\d+|Lesson\s*\d+|Case Study.*)$",
    re.IGNORECASE,
)

# ---------------------------
# Utility: encoding fallbacks
# ---------------------------
def decode_bytes_with_fallback(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

# ---------------------------
# PDF & text extraction
# ---------------------------
def extract_text_from_pdf_fileobj(file_obj) -> str:
    # file_obj: BytesIO or StreamlitUploadedFile
    data = file_obj.read()
    # try pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages)
            if text.strip():
                return text
    except Exception:
        pass
    # fallback to PyPDF2
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass
    # if still nothing, try decode raw bytes as text file
    try:
        return decode_bytes_with_fallback(data)
    except Exception:
        return ""

def extract_text_from_txt_fileobj(file_obj) -> str:
    data = file_obj.read()
    if isinstance(data, bytes):
        return decode_bytes_with_fallback(data)
    return str(data)

# ---------------------------
# Chunking & topic extraction
# ---------------------------
def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:
                sentences = re.split(r"(?<=[.?!])\s+", p)
                s_cur = ""
                for s in sentences:
                    if len(s_cur) + len(s) + 1 <= max_chars:
                        s_cur = (s_cur + " " + s).strip() if s_cur else s
                    else:
                        if s_cur:
                            chunks.append(s_cur)
                        s_cur = s
                if s_cur:
                    chunks.append(s_cur)
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks

def extract_topics_from_chapter_text(ch_text: str, max_topics: int = 40) -> List[str]:
    lines = [l.strip() for l in ch_text.splitlines() if l.strip()]
    topics = []
    for line in lines:
        if HEADING_RE.match(line):
            # clean heading
            h = re.sub(r"[^0-9A-Za-z\s\-\&\:\,\(\)\.]", "", line).strip()
            if h and len(h) > 3:
                topics.append(h)
            if len(topics) >= max_topics:
                break
    # fallback: use first sentences as topics
    if not topics:
        sents = re.split(r"(?<=[.?!])\s+", ch_text)
        topics = [s.strip() for s in sents if 10 < len(s) < 120][:min(10, len(sents))]
    return topics

# ---------------------------
# Book parsing into hierarchy
# ---------------------------
def parse_book_hierarchy(text: str) -> Dict:
    """
    Returns nested dict: { 'parts': { part_title: { 'chapters': { 'Chapter 1': chapter_text } } }, 'chapters_no_part': {...} }
    We'll return a clean structure for UI consumption.
    """
    lines = text.splitlines()
    current_part = None
    current_chapter = None
    structure = {"parts": {}, "chapters_no_part": {}}
    buffer = []

    def flush_chapter():
        nonlocal current_part, current_chapter, buffer
        if current_chapter:
            chap_text = "\n".join(buffer).strip()
            target = structure["chapters_no_part"] if not current_part else structure["parts"].setdefault(current_part, {}).setdefault("chapters", {})
            # ensure chapters container
            if current_part:
                parts_chaps = structure["parts"][current_part].setdefault("chapters", {})
                parts_chaps.setdefault(current_chapter, "")
                parts_chaps[current_chapter] += "\n" + chap_text
            else:
                structure["chapters_no_part"].setdefault(current_chapter, "")
                structure["chapters_no_part"][current_chapter] += "\n" + chap_text
        buffer = []

    for raw in lines:
        line = raw.strip()
        if not line:
            buffer.append("")  # keep paragraph division
            continue
        pmatch = PART_RE.match(line)
        cmatch = CHAPTER_RE.match(line)
        if pmatch:
            # new part -> flush current chapter first
            flush_chapter()
            current_part = pmatch.group(1).strip()
            current_chapter = None
            buffer = []
            continue
        if cmatch:
            # new chapter -> flush previous
            flush_chapter()
            current_chapter = f"Chapter {cmatch.group(2)}"
            buffer = []
            continue
        # accumulate
        buffer.append(line)
    # final flush
    flush_chapter()
    return structure

# ---------------------------
# Persistence helpers
# ---------------------------
def load_local_books_meta() -> dict:
    if os.path.exists(BOOKS_JSON):
        try:
            with open(BOOKS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            try:
                with open(BOOKS_JSON, "r", encoding="latin-1") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}

def save_local_books_meta(meta: dict):
    with open(BOOKS_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ---------------------------
# Ingest book (file object) -> index chunks to chroma and save meta
# ---------------------------
def ingest_book(uploaded_file, book_title: str = None) -> Tuple[bool, str]:
    # uploaded_file: StreamlitUploadedFile or file-like
    # detect extension
    name = book_title or getattr(uploaded_file, "name", "uploaded_book")
    ext = (getattr(uploaded_file, "name", "") or "").lower().split(".")[-1]
    # read text
    text = ""
    try:
        if ext in ("pdf",):
            text = extract_text_from_pdf_fileobj(uploaded_file)
        elif ext in ("txt",):
            text = extract_text_from_txt_fileobj(uploaded_file)
        else:
            # try PDF extractor as fallback
            text = extract_text_from_pdf_fileobj(uploaded_file)
    except Exception as e:
        return False, f"Extraction failed: {e}"

    if not text or len(text.strip()) < 50:
        return False, "Could not extract meaningful text from file."

    # build hierarchy
    hierarchy = parse_book_hierarchy(text)

    # chunk full text and index into chroma
    chunks = chunk_text(text, max_chars=1200)
    doc_ids, docs, metas = [], [], []
    for i, c in enumerate(chunks):
        cid = hashlib.sha1(f"{name}_{i}_{len(c)}".encode()).hexdigest()
        doc_ids.append(cid)
        docs.append(c)
        metas.append({"book": name, "chunk_index": i})
    try:
        embs = embedder.encode(docs)
        # add to chroma
        books_collection.add(ids=doc_ids, documents=docs, embeddings=embs, metadatas=metas)
    except Exception as e:
        return False, f"Embedding/indexing failure: {e}"

    # store metadata (hierarchy and summary)
    meta_all = load_local_books_meta()
    meta_all[name] = {
        "title": name,
        "n_chunks": len(chunks),
        "hierarchy": hierarchy,
        "sample_topics": {
            "parts": {p: extract_topics_from_chapter_text("\n".join(c.get("chapters", {}).values() or []) )[:8] for p, c in hierarchy.get("parts", {}).items()},
            "chapters_no_part": list(hierarchy.get("chapters_no_part", {}).keys())[:8],
        },
    }
    save_local_books_meta(meta_all)
    return True, f"Indexed {len(chunks)} chunks and saved hierarchy."

# ---------------------------
# Posts CSV ingest
# ---------------------------
def ingest_posts_csv_bytes(csv_bytes: bytes) -> Tuple[bool, str]:
    # try multiple decodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc, dtype=str).fillna("")
            break
        except Exception:
            df = None
    if df is None:
        return False, "CSV decoding failed for common encodings."

    if "id" not in df.columns or "content" not in df.columns:
        return False, "CSV must include 'id' and 'content' columns."

    if "profile" not in df.columns:
        df["profile"] = "unknown"

    ids = df["id"].astype(str).tolist()
    docs = df["content"].astype(str).tolist()
    md = [{"profile": p} for p in df["profile"].astype(str).tolist()]

    try:
        # replace collection entirely to keep things clean on ingest
        try:
            chroma_client.delete_collection(POSTS_COLL)
            posts_coll = chroma_client.create_collection(POSTS_COLL)
        except Exception:
            posts_coll = chroma_client.get_or_create_collection(name=POSTS_COLL)
        embeddings = embedder.encode(docs)
        posts_coll.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=md)
    except Exception as e:
        return False, f"Failed to index posts: {e}"
    return True, f"Ingested {len(docs)} posts."

def get_detected_profiles() -> List[str]:
    try:
        metas = posts_collection.get(include=["metadatas"], limit=1000).get("metadatas", [])
        profs = set()
        for m in metas:
            if isinstance(m, dict) and m.get("profile"):
                profs.add(m.get("profile"))
        if not profs:
            return ["Default"]
        return sorted(list(profs))
    except Exception:
        return ["Default"]

# ---------------------------
# Retrieval helpers
# ---------------------------
def retrieve_book_chunks_for_context(query: str, n: int = 5) -> List[str]:
    q_emb = embedder.encode([query])[0]
    res = books_collection.query(query_embeddings=[q_emb], n_results=n)
    docs = res.get("documents", [[]])[0]
    return [d for d in docs if d]

def retrieve_style_examples_for_query(query: str, n: int = 5) -> List[str]:
    if posts_collection.count() == 0:
        return []
    q_emb = embedder.encode([query])[0]
    res = posts_collection.query(query_embeddings=[q_emb], n_results=n)
    docs = res.get("documents", [[]])[0]
    return [d for d in docs if d]

# ---------------------------
# Generation helpers (calls deepseek)
# ---------------------------
PROMPT_SYSTEM = """
You are a Prompt Architect for a digital marketing strategist AI.
Your task is to take a raw 'idea' or 'topic' and turn it into ONE single rich, detailed prompt
that will guide another AI to write a high-quality LinkedIn thought leadership post.

RULES:
- Add context: what angle to take (mistake, myth, strategy shift, efficiency gain).
- Add structure: request Hook → Insight → Framework → Example → CTA.
- Add audience lens: CMO, CEO, CFO, or Director of Digital.
- Ensure uniqueness: each prompt must stand alone.
- Always tie the solution in the prompt to the 5Ws marketing framework.
- Keep the output under 120 words.
"""

PROMPT_5WS = {
    "Who": "Focus on the audience and stakeholders - who benefits, who should act?",
    "What": "Focus on the offering and value proposition - what is it, what changes?",
    "When": "Focus on timing, seasonality, or milestones - when should one act?",
    "Where": "Focus on channels, markets and places - where should the strategy be applied?",
    "Why": "Focus on rationale, business case, and goals - why this approach matters?",
}

def call_generation_system(system_prompt: str, user_text: str) -> str:
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
    )
    return chat.choices[0].message.content

def generate_prompts(idea: str, n: int = 3, use_5ws: bool = True) -> List[Dict]:
    out = []
    if use_5ws and n >= 5:
        for w, focus in PROMPT_5WS.items():
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nFocus on {w}: {focus}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": w, "prompt": res})
        # extras if requested
        extras = n - 5
        for i in range(extras):
            lens = ["CMO", "CEO", "Director of Digital", "CFO"][i % 4]
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nAudience lens: {lens}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": None, "prompt": res})
    else:
        for i in range(n):
            lens = ["CMO", "CEO", "Director of Digital", "CFO"][i % 4]
            user = f"Create a single prompt (<=120 words) from idea:\n\n{idea}\n\nAudience lens: {lens}"
            res = call_generation_system(PROMPT_SYSTEM, user)
            out.append({"w": None, "prompt": res})
    return out

PROFILE_TEMPLATE = """
You are writing LinkedIn-style posts in the voice of: {profile}

Tone:
- Clear, structured, useful.
- Use a hook, offer an insight, provide an actionable framework, show a short example, and end with a light CTA.

Style Hints (do not copy verbatim; use as tonal guidance):
{style_examples_block}

Book / Topic Inspiration (use these to root the post in subject-matter context):
{book_idea_block}

Constraints:
- Keep to ~220 words (shorter is fine).
- Use simple sentences and a professional voice.
"""

def generate_post(query: str, profile_name: str, style_examples: List[str], book_contexts: List[str]) -> str:
    style_examples_block = "\n\n".join(f"- {s[:300]}..." for s in style_examples) if style_examples else "None"
    book_block = "\n\n".join(b[:600] + ("..." if len(b) > 600 else "") for b in book_contexts) if book_contexts else "None"
    system = PROFILE_TEMPLATE.format(profile=profile_name, style_examples_block=style_examples_block, book_idea_block=book_block)
    chat = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Write a LinkedIn post about: {query}"}])
    return chat.choices[0].message.content

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Unified Prompt & Post Generator", page_icon="📚🤖", layout="wide")
st.title("📚🤖 Unified Prompt + Post Generator")

# Sidebar: data ingestion & info
st.sidebar.header("Data Ingest & Storage")
st.sidebar.markdown("Upload books (PDF/TXT) and posts CSV. Encodings are handled automatically. Embeddings stored in `chroma_db_unified`.")

books_meta = load_local_books_meta()

with st.sidebar.expander("📚 Ingest Book (PDF / TXT)", expanded=True):
    uploaded_book = st.file_uploader("Choose a book file", type=["pdf", "txt"], key="book_upload")
    book_title = st.text_input("Optional title (friendly)", value="", key="book_title")
    if uploaded_book is not None:
        if st.button("Ingest Book", key="ingest_book_btn"):
            ok, msg = ingest_book(uploaded_book, book_title.strip() or getattr(uploaded_book, "name", "Uploaded Book"))
            if ok:
                st.success(msg)
                books_meta = load_local_books_meta()
            else:
                st.error(msg)

    st.markdown("**Ingested books:**")
    if books_meta:
        for k, v in books_meta.items():
            n_chunks = v.get("n_chunks", 0)
            st.markdown(f"- **{k}** — {n_chunks} chunks")
        if st.button("Clear local book meta (keeps chroma)", key="clear_book_meta"):
            try:
                os.remove(BOOKS_JSON)
            except Exception:
                pass
            books_meta = {}
            st.experimental_rerun()
    else:
        st.info("No books ingested yet.")

with st.sidebar.expander("📝 Ingest Posts CSV (id,content,profile)", expanded=True):
    uploaded_posts = st.file_uploader("Choose posts CSV", type=["csv"], key="posts_csv")
    if uploaded_posts is not None:
        if st.button("Ingest Posts CSV", key="ingest_posts_btn"):
            ok, msg = ingest_posts_csv_bytes(uploaded_posts.getvalue())
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    try:
        st.markdown(f"**Posts indexed:** {posts_collection.count()}")
    except Exception:
        st.markdown("**Posts indexed:** 0")

# Layout: Tabs for Prompt and Post generators
tab1, tab2 = st.tabs(["Prompt Generator", "Post Generator"])

# ---------------------------
# Prompt Generator Tab
# ---------------------------
with tab1:
    st.header("⚡ Prompt Generator")
    st.markdown("Select book → part → chapter → topic (optional) to seed idea, or paste free-text idea.")
    col1, col2 = st.columns([2, 1])

    with col1:
        idea_text = st.text_area("Idea / Topic (paste chapter/topic text or type an idea):", height=220)
    with col2:
        # book selectors
        book_options = list(books_meta.keys()) if books_meta else []
        selected_book = st.selectbox("Select Book (optional)", options=["None"] + book_options, index=0)
        selected_part = None
        selected_chapter = None
        selected_topic = None
        if selected_book and selected_book != "None":
            hierarchy = books_meta[selected_book]["hierarchy"]
            parts = list(hierarchy.get("parts", {}).keys())
            chapters_no_part = list(hierarchy.get("chapters_no_part", {}).keys())
            # show parts if present
            if parts:
                selected_part = st.selectbox("Select Part (if available)", options=["All Parts"] + parts)
                if selected_part and selected_part != "All Parts":
                    chapter_keys = list(hierarchy["parts"][selected_part].get("chapters", {}).keys())
                    chapter_choices = ["Select Chapter"] + chapter_keys
                    selected_chapter = st.selectbox("Select Chapter", options=chapter_choices)
                    # topics
                    if selected_chapter and selected_chapter != "Select Chapter":
                        chap_text = hierarchy["parts"][selected_part]["chapters"].get(selected_chapter, "")
                        topics = extract_topics_from_chapter_text(chap_text)
                        topic_choices = ["All Topics"] + topics
                        selected_topic = st.selectbox("Select Topic (main headings)", options=topic_choices)
                else:
                    # All parts -> ask for chapter number across book
                    # flatten chapters
                    all_chaps = []
                    for p in parts:
                        all_chaps.extend(list(hierarchy["parts"][p].get("chapters", {}).keys()))
                    all_chaps.extend(chapters_no_part)
                    if all_chaps:
                        selected_chapter = st.selectbox("Select Chapter (All Parts)", options=["Select Chapter"] + all_chaps)
                        if selected_chapter and selected_chapter != "Select Chapter":
                            # locate its chapter text
                            chap_text = ""
                            found = False
                            for p in parts:
                                if selected_chapter in hierarchy["parts"][p].get("chapters", {}):
                                    chap_text = hierarchy["parts"][p]["chapters"][selected_chapter]
                                    found = True
                                    break
                            if not found:
                                chap_text = hierarchy["chapters_no_part"].get(selected_chapter, "")
                            topics = extract_topics_from_chapter_text(chap_text)
                            selected_topic = st.selectbox("Select Topic (main headings)", options=["All Topics"] + topics)
            else:
                # no parts; show chapters
                if chapters_no_part:
                    selected_chapter = st.selectbox("Select Chapter", options=["Select Chapter"] + chapters_no_part)
                    if selected_chapter and selected_chapter != "Select Chapter":
                        chap_text = hierarchy["chapters_no_part"].get(selected_chapter, "")
                        topics = extract_topics_from_chapter_text(chap_text)
                        selected_topic = st.selectbox("Select Topic (main headings)", options=["All Topics"] + topics)

        # prompt controls
        n_prompts = st.slider("How many prompts", 1, 10, 3)
        ensure_5ws = st.checkbox("Ensure 5Ws coverage when >=5 prompts", True)
        gen_btn = st.button("Generate Prompt(s)", key="gen_prompts_btn")

    # build final seed idea if user selected book/chapter/topic
    seed_text = idea_text.strip()
    if not seed_text and selected_book and selected_book != "None" and selected_chapter and selected_chapter != "Select Chapter":
        # choose the chapter/topic text from hierarchy
        hierarchy = books_meta[selected_book]["hierarchy"]
        chapter_text = ""
        if selected_part and selected_part != "All Parts":
            chapter_text = hierarchy["parts"][selected_part]["chapters"].get(selected_chapter, "")
        else:
            # search parts then no-part
            if "parts" in hierarchy:
                for p in hierarchy["parts"].keys():
                    chapter_text = hierarchy["parts"][p]["chapters"].get(selected_chapter, "")
                    if chapter_text:
                        break
            if not chapter_text:
                chapter_text = hierarchy["chapters_no_part"].get(selected_chapter, "")
        if selected_topic and selected_topic != "All Topics":
            # narrow to topic paragraph—best-effort by searching for heading and returning next 2 paragraphs
            lines = chapter_text.splitlines()
            joined = "\n".join(lines)
            idx = joined.find(selected_topic)
            if idx != -1:
                snippet = joined[idx: idx + 1200]
                seed_text = snippet
            else:
                seed_text = chapter_text[:1200]
        else:
            seed_text = chapter_text[:2000]

    if gen_btn:
        if not seed_text:
            st.error("Please provide an idea text OR select a book+chapter/topic to seed from.")
        else:
            with st.spinner("Generating prompts..."):
                prompts_out = generate_prompts(seed_text, n=n_prompts, use_5ws=ensure_5ws)
            st.success("Prompts generated!")
            for i, p in enumerate(prompts_out):
                label = f"[{p['w']}]" if p["w"] else f"Prompt {i+1}"
                st.subheader(label)
                st.code(p["prompt"])
            # download
            joined = "\n\n".join((f"{('[%s]'%p['w']) if p['w'] else ''}\n{p['prompt']}") for p in prompts_out)
            st.download_button("Download prompts (.txt)", joined, file_name="generated_prompts.txt")

# ---------------------------
# Post Generator Tab
# ---------------------------
with tab2:
    st.header("✍️ Post Generator")
    st.markdown("Choose a profile (auto from CSV or custom), choose book context (optional), and generate posts.")
    q_col, opt_col = st.columns([2, 1])

    with q_col:
        post_query = st.text_input("Post topic / headline:", "")
        k_examples = st.slider("Number of style examples to use", 1, 10, 5)
    with opt_col:
        profiles = get_detected_profiles()
        profile_sel = st.selectbox("Profile voice (from CSV)", options=["Auto"] + profiles + ["Custom"])
        custom_profile = ""
        if profile_sel == "Custom":
            custom_profile = st.text_input("Custom profile name (tone):", value="Freelance Marketer")
        # book context picker
        book_opt = st.selectbox("Use Book Context (optional)", options=["None"] + (list(books_meta.keys()) if books_meta else []))
        book_context_texts = []
        if book_opt and book_opt != "None":
            hierarchy = books_meta[book_opt]["hierarchy"]
            parts = list(hierarchy.get("parts", {}).keys())
            chapters_no_part = list(hierarchy.get("chapters_no_part", {}).keys())
            chapter_selected = None
            if parts:
                p_choice = st.selectbox("Part (optional)", options=["All Parts"] + parts)
                if p_choice and p_choice != "All Parts":
                    chapter_selected = st.selectbox("Chapter", options=["Select Chapter"] + list(hierarchy["parts"][p_choice].get("chapters", {}).keys()))
                    if chapter_selected and chapter_selected != "Select Chapter":
                        book_context_texts = [hierarchy["parts"][p_choice]["chapters"][chapter_selected]]
                else:
                    # choose chapter across all
                    all_chaps = []
                    for p in parts:
                        all_chaps.extend(list(hierarchy["parts"][p].get("chapters", {}).keys()))
                    all_chaps.extend(chapters_no_part)
                    if all_chaps:
                        chapter_selected = st.selectbox("Chapter (All Parts)", options=["Select Chapter"] + all_chaps)
                        if chapter_selected and chapter_selected != "Select Chapter":
                            # find it
                            found = False
                            for p in parts:
                                if chapter_selected in hierarchy["parts"][p].get("chapters", {}):
                                    book_context_texts = [hierarchy["parts"][p]["chapters"][chapter_selected]]
                                    found = True
                                    break
                            if not found:
                                book_context_texts = [hierarchy["chapters_no_part"].get(chapter_selected, "")]
            else:
                if chapters_no_part:
                    chapter_selected = st.selectbox("Chapter", options=["Select Chapter"] + chapters_no_part)
                    if chapter_selected and chapter_selected != "Select Chapter":
                        book_context_texts = [hierarchy["chapters_no_part"].get(chapter_selected, "")]

        temp = st.slider("Creativity (temperature proxy)", 0.0, 1.0, 0.7)

        gen_post_btn = st.button("Generate Post", key="gen_post_btn")

    if gen_post_btn:
        if not post_query.strip():
            st.error("Please enter a post topic / headline.")
        else:
            profile_name = custom_profile if profile_sel == "Custom" else (profile_sel if profile_sel != "Auto" else (profiles[0] if profiles else "Default"))
            with st.spinner("Retrieving style examples and generating post..."):
                style_examples = retrieve_style_examples_for_query(post_query, n=k_examples)
                # if no book_context_texts but book_opt selected, retrieve related chunks
                if not book_context_texts and book_opt and book_opt != "None":
                    book_context_texts = retrieve_book_chunks_for_context(post_query, n=3)
                generated = generate_post(post_query, profile_name, style_examples, book_context_texts)
            st.subheader("Generated Post")
            st.write(generated)
            if style_examples:
                with st.expander("Retrieved style examples (tone guidance)"):
                    for s in style_examples:
                        st.markdown(f"- {s[:400]}...")
            if book_context_texts:
                with st.expander("Book contexts used"):
                    for b in book_context_texts:
                        st.markdown(f"- {b[:400]}...")
            st.download_button("Download post (.txt)", generated, file_name="generated_post.txt")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("**Notes:** Embeddings & indices are stored in `chroma_db_unified`. Uploaded books and posts are indexed for fast retrieval. Encoding fallbacks are implemented for PDFs and CSVs. Model used for generation: `deepseek/deepseek-r1:free`.")
