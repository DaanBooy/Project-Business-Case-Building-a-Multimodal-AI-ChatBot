# agent.py — imports
from pathlib import Path
import os
from typing import List, Dict

# Chroma + embeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Self-Query retriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# LLM for routing / self-query / QA 
from langchain_openai import ChatOpenAI           

# Compression to trim context
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Simple QA chain bits
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Session-scoped memory
from collections import defaultdict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Hardiness zone lookup tool
import os, re, json, time
from typing import Optional
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from collections import Counter
import requests

# PDF Tool
from fpdf import FPDF
from datetime import datetime

# Agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor


# --- Env / keys (works locally and on Hugging Face Spaces) ---
import os

try:
    # Optional: for local dev only. On Spaces there is no .env and this is a no-op.
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Mapping LangSmith key to what LangChain expects.
# (Spaces secret created: LANGSMITH_API_KEY. LangChain expects LANGCHAIN_API_KEY.)
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

# If a LangSmith key is present, ensure tracing defaults are on.
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "GrowGuide QA")

# Sanity warnings (don’t crash; tools that need the key just won’t run)
for var in ["OPENAI_API_KEY", "SERPAPI_API_KEY"]:
    if not os.getenv(var):
        print(f"⚠️ {var} is not set; related features may be disabled.")


# --- Load persisted Chroma DB ---
DB_DIR = os.getenv("DB_DIR") or str((Path(__file__).parent / "chroma_growguide").resolve())
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fail fast if the DB folder is missing
if not Path(DB_DIR).exists():
    raise RuntimeError(
        f"Chroma DB not found at {DB_DIR}. Ensure 'chroma_growguide' is committed to the Space."
    )

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}  # Spaces CPU
)

# Open existing DB
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

# --- Self-Query Retriever over Chroma ---
llm_routing = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Describing metadata so the retriever knows what it can filter on
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="Which section the chunk comes from: 'fruits', 'vegetable_planner', or 'vegetable_list'.",
        type="string",
    ),
    AttributeInfo(
        name="crop",
        description="Vegetable crop name for encyclopedia entries (e.g., onion, tomato, pea).",
        type="string",
    ),
    AttributeInfo(
        name="topics",
        description="Comma-separated keywords inside a vegetable crop entry (yield, planting, conditions, soil, care, companions, harvest, storage tips).",
        type="string",
    ),
    AttributeInfo(
        name="zones",
        description="Planner zone label like '3 and 4', '5 and 6', or '7, 8 and 9+'.",
        type="string",
    ),
    AttributeInfo(
        name="month",
        description="Planner month name when present (e.g., March, April).",
        type="string",
    ),
]

# Brief description of what’s in the documents / dataset
document_contents = (
    "A home-growing guide covering fruits, a vegetable planting/harvest planner by USDA zones and months, "
    "and a per-vegetable encyclopedia with labeled fields such as planting, care, companions, storage, soil."
)

# Building the self-query retriever on top of existing vectordb
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm_routing,
    vectorstore=vectordb,
    document_contents=document_contents,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    search_kwargs={"k": 6},             # final number of docs returned
)

# --- RAG chain using self query retriever ---
def format_docs(docs):
    parts = []
    for d in docs:
        m = d.metadata
        tag = f"[{m.get('category')}"
        if m.get("crop"):   tag += f"|{m['crop']}"
        if m.get("zones"):  tag += f"|zones:{m['zones']}"
        if m.get("month"):  tag += f"|month:{m['month']}"
        tag += "]"
        parts.append(f"{tag} {d.page_content}")
    return "\n\n".join(parts)

# prompt
qa_prompt = PromptTemplate.from_template(
    """You are a helpful home-growing assistant.

SCOPE
• In-scope topics: home fruit & vegetable growing, hardiness climate zones, and growing/seasonal planning (including month-by-month tasks).
• If the QUESTION is outside this scope (e.g., gaming, finance, travel, general tech, etc.) OR the CONTEXT does not relate to those topics, do NOT answer.
  Instead reply something along the lines of:
  "Sorry, this isn’t my expertise. I’m focused on home fruit & vegetable growing, hardiness climate zones, and planning. I can help with things like crops, soil, seasons, zones, or create a growing planner."

RULES
• Use ONLY the provided CONTEXT. If the answer is not in the context, say you don’t know.
• Keep it brief and focused: 1–4 sentences or a short bullet list that directly answers the question.
• Do not add unrelated background or long how-tos unless asked.
• NOT necessary but If you think it is helpful, offer to provided more information, something alone the lines of: “I can share step-by-step details or a month-by-month plan if you’d like.”
• If the CONTEXT includes a crop, month, or zone, tailor the answer to that; otherwise keep it general.
• Do NOT mention “context” or include citations in your answer.

QUESTION: {question}

CONTEXT:
{context}
"""
)

# LLM to generate the answer
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# add LangSmith run names/tags
sq_retriever_traced = self_query_retriever.with_config(
    {"run_name": "self_query_retriever", "tags": ["retriever","self_query","chroma"]}
)

rag_chain = (
    {"context": sq_retriever_traced | format_docs, "question": RunnablePassthrough()}
    | qa_prompt
    | llm_answer
    | StrOutputParser()
).with_config({"run_name": "rag_qa_chain", "tags": ["qa","rag","gpt-4o-mini"]})

# --- Adding Tools > memory, zone lookup and page reader, pdf creation ---
# Session memory
# In-process, per-session message storage
_SESSION_STORES = defaultdict(InMemoryChatMessageHistory)

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return/create a chat history bucket for this session."""
    return _SESSION_STORES[session_id]

# Helper to wrap Runnable (QA chain, Agent, etc.) with session memory
def with_session_memory(runnable, *, input_key: str = "question", history_key: str = "history"):
    """
    Wrap a Runnable so it remembers past turns per session.
    - input_key: the key in the invoke(...) dict that carries the user's new message
    - history_key: the key that the prompt/agent expects to receive past messages under
    """
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
    )

# USDA / World hardiness zone lookup (SerpAPI, Plantmaps-biased)
# Stable English results so titles/snippets are predictable
_serp = SerpAPIWrapper(params={"engine": "google", "hl": "en", "gl": "us", "num": 10})

def _first_plantmaps(organic_results):
    """Return first Plantmaps result url if present, else None."""
    for r in organic_results or []:
        url = (r.get("link") or "").strip()
        if "plantmaps.com" in url:
            return url
    return None

@tool("find_hardiness_source", return_direct=False)
def find_hardiness_source(location: str) -> str:
    """
    Return the best page to read the hardiness zone for a location.
    Prefers Plantmaps; falls back to the top Google result for 'hardiness zone <location>'.
    Output: JSON string with {location, best_source, note}.
    """
    if not location or not location.strip():
        return json.dumps({"error": "location required"})

    # 1) Plantmaps-biased query
    q_pm = f"site:plantmaps.com {location} hardiness zone"
    try:
        res_pm = _serp.results(q_pm)
    except Exception as e:
        return json.dumps({"location": location, "best_source": None,
                           "note": f"search error (plantmaps query): {e}"})

    best = _first_plantmaps(res_pm.get("organic_results"))
    if best:
        return json.dumps({"location": location, "best_source": best,
                           "note": "Plantmaps result chosen."})

    # 2) Fallback: generic query
    q_generic = f"hardiness zone {location}"
    try:
        res_generic = _serp.results(q_generic)
    except Exception as e:
        return json.dumps({"location": location, "best_source": None,
                           "note": f"search error (generic query): {e}"})

    # pick the very first organic result if available
    org = (res_generic.get("organic_results") or [])
    best = (org[0].get("link") if org else None)
    return json.dumps({"location": location, "best_source": best,
                       "note": "Generic result chosen." if best else "No results found."})

# Read Plantmaps page and extract the zone (with 403 fallback)
# token like "Zone 8a" / "zone 8" etc.
_ZONE_RE = re.compile(r"\bZone\s*([0-9]{1,2}[ab]?)\b", re.I)

# build a set of fuzzy tokens from the user's location to match rows/lines
def _loc_tokens(location: str):
    s = location.lower()
    # split on commas and spaces, keep tokens with letters
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    # drop very short noise like “us”, “the”
    parts = [p for p in parts if len(p) > 2]
    return set(parts)

def _pick_nearest_zone(lines: list[str], location: str) -> Optional[str]:
    tokens = _loc_tokens(location)
    best_idx, best_zone = None, None

    # 1) exact “City” line match first (common on Plantmaps country pages)
    for i, line in enumerate(lines):
        l = line.lower()
        # a row often looks like: "Berlin Zone 8a: -12.2°C to -9.4°C"
        if all(t in l for t in tokens):
            m_here = _ZONE_RE.search(line)
            if m_here:
                return m_here.group(1).lower()
            # if the zone is on the next line (sometimes), grab it
            if i + 1 < len(lines):
                m_next = _ZONE_RE.search(lines[i + 1])
                if m_next:
                    return m_next.group(1).lower()
            # otherwise remember index, we’ll expand search a bit
            if best_idx is None:
                best_idx = i

    # 2) if no direct hit, search a window around the first approximate hit
    if best_idx is not None:
        for j in range(max(0, best_idx - 3), min(len(lines), best_idx + 4)):
            m = _ZONE_RE.search(lines[j])
            if m:
                return m.group(1).lower()

    # 3) last resort: first zone anywhere on the page (not ideal, but better than nothing)
    for line in lines:
        m = _ZONE_RE.search(line)
        if m:
            return m.group(1).lower()

    return None

def _fetch_text_with_fallback(url: str) -> tuple[Optional[str], str]:
    """Fetch HTML or readable text from URL.
    Returns (text, method_used). Falls back to r.jina.ai on 403/other failures.
    """
    headers = {
        # very “real” browser headers help reduce 403s
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200 and r.text:
            return r.text, "direct"
        # A tiny delay sometimes helps before retrying through the proxy
    except Exception:
        pass

    time.sleep(0.3)
    # Proxy reader that returns a plaintext rendering of the page
    proxy_url = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
    try:
        r2 = requests.get(proxy_url, headers={"User-Agent": headers["User-Agent"]}, timeout=12)
        if r2.status_code == 200 and r2.text:
            return r2.text, "r.jina.ai"
    except Exception:
        pass

    return None, "failed"

@tool("read_zone_from_url", return_direct=False)
def read_zone_from_url(url: str, location: Optional[str] = None) -> str:
    """
    Fetch a Plantmaps (or similar) page and extract a single hardiness zone.
    - Tries a normal fetch with realistic headers.
    - On 403/other failure, falls back to a plaintext proxy (r.jina.ai).
    - Uses the provided `location` to pick the right row on city lists.
    Returns a JSON string: {"zone": "8a", "source": url, "method": "...", "note": "..."}.
    """
    if not url:
        return json.dumps({"zone": None, "source": None, "method": None, "note": "No URL provided"})

    text, method = _fetch_text_with_fallback(url)
    if not text:
        return json.dumps({"zone": None, "source": url, "method": None, "note": "Fetch failed"})

    # Split to lines and normalize
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    zone = _pick_nearest_zone(lines, location or "")

    note = "Parsed from Plantmaps page via direct fetch." if method == "direct" \
        else "Parsed from Plantmaps page via r.jina.ai text proxy."

    return json.dumps({
        "zone": zone,
        "source": url,
        "note": note if zone else note + " No explicit zone token found."
    })

# PDF TOOL config.
PLANNER_OUTDIR = os.getenv("PLANNER_OUTDIR", "/tmp")  # writable on Spaces

# Safety net: force latin-1 safe text (replaces unsupported chars)
def latin1(s: str) -> str:
    return "" if s is None else str(s).encode("latin-1", "replace").decode("latin-1")

# Months in display order
MONTHS_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

class PlannerPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        # Title (single line, explicit width)
        self.set_x(self.l_margin)
        self.cell(self.w - self.l_margin - self.r_margin, 10,
                  getattr(self, "title", "Home Growing Planner"),
                  new_x="LMARGIN", new_y="NEXT")

        subtitle = getattr(self, "subtitle", "")
        if subtitle:
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(80, 80, 80)
            # Wrap subtitle with explicit width
            self.multi_cell(self.w - self.l_margin - self.r_margin, 6, subtitle)
            self.set_text_color(0, 0, 0)

        self.ln(2)
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120,120,120)
        # Footer: explicit width + centered
        self.cell(self.w - self.l_margin - self.r_margin, 10,
                  f"Generated {datetime.now().strftime('%Y-%m-%d')}  -  Page {self.page_no()}",
                  align="C")
        self.set_text_color(0,0,0)

def _order_months(keys: List[str]) -> List[str]:
    order = {m:i for i,m in enumerate(MONTHS_ORDER)}
    return sorted(keys, key=lambda k: order.get(k, 999))

def make_planner_pdf(
    path: str,
    *,
    location: str,
    zone: Optional[str],
    tasks_by_month: Dict[str, List[str]],
    crops: Optional[List[str]] = None,
    notes: str = ""
) -> Dict[str, object]:
    """
    Create a printable growing planner PDF.
    - location: free text (e.g., "Berlin, Germany")
    - zone: e.g., "8a" (can be None)
    - tasks_by_month: {"March": ["Start tomatoes indoors", ...], ...}
    - crops: optional ["tomato","onion",...]
    - notes: optional free text
    Returns: {"path": path, "pages": int, "bytes": int}
    """
    pdf = PlannerPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    subtitle_bits = [latin1(f"Location: {location}")]
    if zone:
        subtitle_bits.append(latin1(f"Zone: {zone}"))
    if crops:
        subtitle_bits.append(latin1("Crops: " + ", ".join(crops)))
    pdf.title = "Home Growing Planner"
    pdf.subtitle = "  -  ".join(subtitle_bits)

    pdf.add_page()

    # Overview (only if notes are provided)
    if notes:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_x(pdf.l_margin)
        pdf.cell(pdf.w - pdf.l_margin - pdf.r_margin, 8, "Overview",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, latin1(notes))
        pdf.ln(2)

    # Month-by-month heading
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.w - pdf.l_margin - pdf.r_margin, 8, "Month-by-Month",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    # Define months from keys (ordered)
    months = _order_months(list(tasks_by_month.keys()))

    # Month sections
    for m in months:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 7, latin1(m))

        pdf.set_font("Helvetica", "", 11)
        rows = tasks_by_month.get(m, [])
        if not rows:
            pdf.set_text_color(120,120,120)
            pdf.set_x(pdf.l_margin)
            pdf.cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, "No specific tasks.",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0,0,0)
        else:
            for task in rows:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, latin1(f"- {task}"))
        pdf.ln(1)

    # Ensure an output directory exists (for Spaces this is safe)
    outdir = os.path.dirname(path) or PLANNER_OUTDIR
    os.makedirs(outdir, exist_ok=True)

    pdf.output(path)
    abs_path = os.path.abspath(path)
    return {"path": abs_path, "pages": pdf.page_no(), "bytes": os.path.getsize(abs_path)}


# how to call the tool, will be used by agent later.
@tool("make_planner_pdf", return_direct=False)
def make_planner_pdf_tool(payload_json: str) -> str:
    """Create a printable growing planner PDF.

    Expects a JSON string with keys:
      - location: str
      - zone: str | null
      - tasks_by_month: dict[str, list[str]]
      - crops: list[str] | null
      - notes: str
      - out_path: optional str (directory or full file path)

    Returns:
      A JSON string: {"path": "...", "pages": int, "bytes": int}
    """
    import json
    data = json.loads(payload_json)

    # default into /tmp (or PLANNER_OUTDIR) if no path provided
    out_dir = data.get("out_path")
    if out_dir:
        # If user supplied a full file path, keep it; otherwise treat as directory.
        if out_dir.endswith(".pdf"):
            out_path = out_dir
        else:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"planner_{int(time.time())}.pdf")
    else:
        out_path = os.path.join(PLANNER_OUTDIR, f"planner_{int(time.time())}.pdf")

    result = make_planner_pdf(
        path=out_path,
        location=data.get("location", ""),
        zone=data.get("zone"),
        tasks_by_month=data.get("tasks_by_month"),
        crops=data.get("crops") or None,
        notes=data.get("notes", ""),
    )
    return json.dumps(result)



# --- QA Agent with tools (Self-Query RAG for Q&A + Zone lookup + PDF) ---
# Wrap existing RAG chain as a tool so the agent uses it for answers.
#    NOTE: rag_chain must already be defined as in earlier cell
#    (it uses self_query_retriever under the hood).
@tool("growguide_answer", return_direct=False)
def growguide_answer(question: str) -> str:
    """
    Answer the user's question using the Grow Guide RAG QA chain.
    (This chain already uses the self-query retriever.)
    Returns a concise, step-by-step answer.
    """
    return rag_chain.invoke(question)

# 1) Expose the retriever as a raw-search tool for planner building.
@tool("search_growguide", return_direct=False)
def search_growguide(query: str) -> str:
    """
    Retrieve up to 6 relevant chunks from the Grow Guide using the Self-Query retriever.
    Returns a JSON list of items: [{"content": "...", "meta": {...}}, ...]
    Use this to extract month-specific tasks, crop tips, etc., e.g., when building a PDF planner.
    """
    hits = self_query_retriever.get_relevant_documents(query) or []
    out = [{"content": h.page_content, "meta": h.metadata} for h in hits[:6]]
    return json.dumps(out)

# 2) Agent prompt: route to the right tool
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful home-growing assistant.\n"
     "TOOLS:\n"
     "• growguide_answer — Use for most Q&A. It runs the RAG QA chain (self-query retriever) and returns a ready answer.\n"
     "• search_growguide — Use when you need raw snippets to assemble tasks_by_month for a PDF planner.\n"
     "• find_hardiness_source + read_zone_from_url — Use to determine a location's hardiness zone.\n\n"
     "Guidance:\n"
     "1) If the user asks for their hardiness zone: call 'find_hardiness_source' with the location, then "
     "'read_zone_from_url' with that URL (and the location) to extract the zone; remember it in this chat.\n"
     "2) For normal growing questions: call 'growguide_answer' with the user's question (include any known zone/crop/months).\n"
     "3) If a full planner/PDF is requested: call 'search_growguide' to fetch snippets, build tasks_by_month (month -> [task,...]), "
     "then call 'make_planner_pdf' with location, zone (if known), tasks_by_month, optional crops, and notes. Use ASCII/latin-1 text.\n"
     "4) Keep answers focused and brief. Directly answer the user's question **only**. Prefer 1–4 sentences or a short bullet list. "
     "Avoid extra detail unless asked; instead, offer a follow-up like: "
     "\"I can share more detail (e.g., step-by-step or month-by-month tasks) if you’d like.\"\n"
     "5) Stay in scope: home gardening of fruits/vegetables, hardiness climate zones, and printable planning PDFs. "
     "If a request is outside this scope (e.g., gaming, finance, unrelated tech), reply briefly: "
     "\"Sorry, that’s outside my expertise. I can help with home fruit & vegetable growing, climate zones, and personalized planning.\""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# 3) LLM + tools
llm_tools = [
    growguide_answer,        # <- preferred for Q&A (uses self-query retriever via rag_chain)
    search_growguide,        # <- raw chunks for planner building (also uses self-query retriever)
    find_hardiness_source,   # SerpAPI step 1 (best source URL as JSON)
    read_zone_from_url,      # SerpAPI step 2 (parse page to get zone)
    make_planner_pdf_tool,   # PDF generator tool
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_openai_tools_agent(llm, llm_tools, agent_prompt)
agent_exec = AgentExecutor(
    agent=agent,
    tools=llm_tools,
    verbose=False,
    max_iterations=8,
    handle_parsing_errors=True,
).with_config({"run_name": "growguide_tools_agent", "tags": ["agent","tools","rag"]})

# 4) Session memory wrapper that was already built
qa_agent = with_session_memory(agent_exec, input_key="input", history_key="chat_history")

def run_agent(question: str, session_id: str):
    """Single entry point the app can call."""
    cfg = {"configurable": {"session_id": session_id}}
    out = qa_agent.invoke({"input": question}, config=cfg)
    # AgentExecutor returns {"output": "..."} by default
    return out.get("output", out)

