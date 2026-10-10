import base64
import http
import os
import sys

# Store __pycache__ inside the cache/ folder to keep the root directory clean
sys.pycache_prefix = os.path.join(os.path.dirname(__file__), "cache", "__pycache__")

import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import uvicorn
from collections import defaultdict
from typing import Optional
from intent_router import classify_query_intent

# FastAPI is the core framework used to build our web API
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

# CORSMiddleware allows our frontend (HTML files) to communicate with this backend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from jose import jwt

# Pydantic is used to define the structure of incoming data (like JSON requests)
from pydantic import BaseModel
from sqlalchemy import select, text

# SQLAlchemy components for interacting with our PostgreSQL database
from sqlalchemy.ext.asyncio import AsyncSession

# StaticFiles already imported above
# Import our custom database configurations and models
from database import Base, Document, DocumentChunk, ProcessingReport, engine, get_db, get_neo4j
from query_neo4j import fetch_all_problems_by_topic, fetch_problems_by_topic_graph
from utils import (
    PREREQUISITE_MAP,
    add_prerequisite_edges,
    delete_all_images_from_cloudinary,
    delete_document_images_from_cloudinary,
    describe_uploaded_image,
    execute_neo4j_pyq_search,
    extract_pyq_questions,
    extract_pyq_structured,
    generate_answer,
    generate_answer_stream,
    get_embedding,
    hybrid_search_rrf,
    map_pyq_structured_to_kg,
    map_questions_to_kg,
    process_pyq_visuals,
    upload_question_image_to_cloudinary,
)

# Application Setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts.
    It connects to PostgreSQL and ensures all necessary tables exist.
    """
    async with engine.begin() as conn:
        # We must enable the 'vector' extension in PostgreSQL before creating tables
        # that use the Vector data type (like our DocumentChunk table).
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables defined in our SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)

        # Add generated column and GIN index for TSVector full text search
        await conn.execute(
            text("""
            ALTER TABLE document_chunks 
            ADD COLUMN IF NOT EXISTS tsv_content TSVECTOR 
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """)
        )
        await conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN(tsv_content)
        """)
        )

    print("Database startup complete: Tables verified.")

    # Create Neo4j indexes for the new GraphRAG schema
    try:
        neo4j_driver = get_neo4j()
        with neo4j_driver.session() as _ns:
            _ns.run(
                "CREATE INDEX course_idx IF NOT EXISTS FOR (c:Course) ON (c.code, c.dept, c.year)"
            )
            _ns.run(
                "CREATE INDEX co_idx IF NOT EXISTS FOR (co:CourseOutcome) ON (co.id, co.course_code)"
            )
            _ns.run(
                "CREATE INDEX sv_idx IF NOT EXISTS FOR (sv:SyllabusVersion) ON (sv.dept, sv.year)"
            )
            _ns.run(
                "CREATE INDEX page_idx IF NOT EXISTS FOR (p:Page) ON (p.page_num, p.syllabus_version_id)"
            )
            _ns.run(
                "CREATE INDEX dept_idx IF NOT EXISTS FOR (d:Department) ON (d.name)"
            )
            _ns.run(
                "CREATE FULLTEXT INDEX course_name_ft IF NOT EXISTS FOR (n:Course) ON EACH [n.name, n.code, n.title]"
            )
        print("Neo4j indexes created/verified.")
    except Exception as e:
        print(f"Neo4j index setup warning: {e}")

    # Seed Neo4j prerequisite graph (legacy PYQ graph)
    try:
        add_prerequisite_edges(neo4j_driver, PREREQUISITE_MAP)
        print("Neo4j Prerequisite Graph seeded.")
    except Exception as e:
        print(f"Failed to seed prereqs: {e}")

    yield


app = FastAPI(
    title="M.A.C.H. - Multimodel Academic Cognitive Hub",
    description="A Fine-tuned Graph-RAG framework for Campus Knowledge Systems (MACH AI)",
    lifespan=lifespan,
)

from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add GZip compression for large JSON payloads
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure Cross-Origin Resource Sharing (CORS).
# This allows our local HTML files (opened directly in the browser) to send requests to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=False,  # False because we use Bearer tokens, not cookies
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
    expose_headers=["X-Response-Type", "X-Message-ID"],  # Allow frontend to read custom headers
)


@app.middleware("http")
async def custom_logging(request: Request, call_next):
    response = await call_next(request)
    if response.status_code != 304:
        # Replicate Uvicorn's default access log format, but ignore 304 Not Modified
        try:
            phrase = http.HTTPStatus(response.status_code).phrase
        except ValueError:
            phrase = ""
        client_host = request.client.host if request.client else "127.0.0.1"
        client_port = request.client.port if request.client else 0
        http_version = request.scope.get("http_version", "1.1")
        print(
            f'INFO:     {client_host}:{client_port} - "{request.method} {request.url.path} HTTP/{http_version}" {response.status_code} {phrase}'
        )
    return response


# Mount the static directory so the frontend can retrieve images
os.makedirs("uploads/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Serve the frontend at /public/
# The frontend directory lives one level up from backend/
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount(
        "/public", StaticFiles(directory=_frontend_dir, html=True), name="frontend"
    )


# Route: GET / — Convenience redirect to the frontend login page


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/public/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Attempt to resolve the frontend favicon path
    favicon_path = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "favicon.ico"
    )
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)


# Auth Configuration

_JWT_SECRET = settings.jwt_secret
_JWT_ALGORITHM = settings.jwt_algorithm
_JWT_EXPIRE = settings.jwt_expire_minutes

# Demo credentials — matches the hints shown on the login page
_USERS = {
    "student": {"password": "student123", "role": "student"},
    "admin": {"password": "admin123", "role": "admin"},
}


def _create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=_JWT_EXPIRE),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


# Route: POST /login — JWT Authentication

from fastapi import Form as FastForm


@app.post("/login")
async def login(username: str = FastForm(...), password: str = FastForm(...)):
    """
    Validates credentials against the hardcoded demo user dict.
    Returns a signed JWT access token on success.
    The frontend decodes the JWT to extract the user's role.
    """
    user = _USERS.get(username.lower())
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    token = _create_token(username.lower(), user["role"])
    return {"access_token": token, "token_type": "bearer"}


# Route: POST /feedback/{message_id} — Student Thumbs Up/Down


class FeedbackRequest(BaseModel):
    score: int  # 1 = helpful, -1 = not helpful


@app.post("/feedback/{message_id}")
async def submit_feedback(message_id: str, body: FeedbackRequest):
    """
    Receives thumbs-up/down feedback from the student chat UI.
    Logs it for now; can be persisted to a DB table in future.
    """
    print(f"[Feedback] message_id={message_id}  score={body.score}")
    return {
        "message": "Feedback recorded",
        "message_id": message_id,
        "score": body.score,
    }


# Route: /chat — Handles Student Questions

import httpx

from utils import OLLAMA_BASE_URL, OLLAMA_MODEL





def execute_graph_prereq_query(neo4j_driver, question: str) -> list:
    """Extracts course name/code from question and finds prereqs."""
    # Rough extraction for demonstration (assumes course is in the query)
    words = question.lower().split()
    course_hint = next(
        (
            w
            for w in words
            if len(w) > 3 and w not in ["what", "are", "the", "prerequisites", "for"]
        ),
        "",
    )

    with neo4j_driver.session() as session:
        records = session.run(
            """
            MATCH (target:Course)
            WHERE toLower(target.name) CONTAINS toLower($course_name) 
               OR toLower(target.code) CONTAINS toLower($course_name)
            MATCH path = (target)-[:REQUIRES*1..5]->(prereq:Course)
            RETURN target.name AS course, collect(DISTINCT prereq.name) AS all_prerequisites
            LIMIT 10
        """,
            course_name=course_hint,
        ).data()
    return records


def execute_graph_co_query(neo4j_driver, question: str) -> list:
    """Extracts CO id and queries the graph."""
    import re

    co_match = re.search(r"co\d+", question.lower())
    co_id = co_match.group(0).upper() if co_match else "CO1"

    with neo4j_driver.session() as session:
        records = session.run(
            """
            MATCH (q:Question)-[:MAPPED_TO_CO]->(co:CourseOutcome {id: $co_id})
                  -[:BELONGS_TO]->(c:Course)
            RETURN q.text AS question, q.btl AS btl, q.marks AS marks, q.has_figure AS has_figure, c.code AS course_code
            LIMIT 20
        """,
            co_id=co_id,
        ).data()
    return records


def execute_graph_syllabus_query(neo4j_driver, question: str) -> dict:
    """
    Returns structured syllabus query result:
      {"match_type": "exact",        "records": [...]}
      {"match_type": "similar_list", "courses": [...]}
      {"match_type": "none",         "records": []}
    Unit records are sorted ascending by unit_number.
    """
    STOP = {
        "get", "me", "the", "syllabus", "for", "course", "topics", "units", "unit",
        "what", "is", "give", "show", "list", "provide", "all", "of", "about",
        "in", "please", "and", "a", "an",
    }
    words = [w.strip(".,!?'\"") for w in question.lower().split()
             if len(w) > 1 and w.lower() not in STOP]
    phrase = " ".join(words).strip()

    with neo4j_driver.session() as session:
        # 1. Exact match on code or full name
        if phrase:
            exact = session.run("""
                MATCH (c:Course)
                WHERE toLower(c.code) = toLower($sp) OR toLower(c.name) = toLower($sp)
                OPTIONAL MATCH (c)-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
                WITH c, u, t ORDER BY toInteger(last(split(t.topic_id, '_t'))) ASC
                WITH c, u.number AS unit_num,
                     coalesce(u.title, 'Unit ' + toString(u.number)) AS unit_title,
                     collect(t.name) AS topics
                WHERE size(topics) > 0
                RETURN c.code AS course_code, c.name AS course_name,
                       toInteger(unit_num) AS unit_number, unit_title, topics
                ORDER BY unit_number ASC
            """, sp=phrase).data()
            if exact:
                return {"match_type": "exact", "records": exact}

        # 2. Keyword match
        candidates = []
        if words:
            candidates = session.run("""
                MATCH (c:Course)
                WHERE all(word IN $words WHERE toLower(c.name) CONTAINS word
                          OR toLower(c.code) CONTAINS word)
                RETURN c.code AS course_code, c.name AS course_name
                ORDER BY size(c.name) ASC LIMIT 10
            """, words=words).data()

        if len(candidates) == 1:
            single = session.run("""
                MATCH (c:Course {code: $c_code})-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
                WITH c, u, t ORDER BY toInteger(last(split(t.topic_id, '_t'))) ASC
                WITH c, u.number AS unit_num,
                     coalesce(u.title, 'Unit ' + toString(u.number)) AS unit_title,
                     collect(t.name) AS topics
                WHERE size(topics) > 0
                RETURN c.code AS course_code, c.name AS course_name,
                       toInteger(unit_num) AS unit_number, unit_title, topics
                ORDER BY unit_number ASC
            """, c_code=candidates[0]["course_code"]).data()
            if single:
                return {"match_type": "exact", "records": single}

        if len(candidates) > 1:
            return {"match_type": "similar_list", "courses": candidates}

        # 3. Topic-level fallback
        if words:
            topic_records = session.run("""
                MATCH (c:Course)-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
                WHERE all(word IN $words WHERE toLower(t.name) CONTAINS word)
                WITH c, u.number AS unit_num,
                     coalesce(u.title, 'Unit ' + toString(u.number)) AS unit_title,
                     collect(DISTINCT t.name) AS topics
                WHERE size(topics) > 0
                RETURN c.code AS course_code, c.name AS course_name,
                       toInteger(unit_num) AS unit_number, unit_title, topics
                ORDER BY c.code, unit_num ASC LIMIT 15
            """, words=words).data()
            if topic_records:
                return {"match_type": "exact", "records": topic_records}

    return {"match_type": "none", "records": []}


class ChatRequest(BaseModel):
    """Defines the expected JSON structure when a student asks a question."""

    message: str
    session_id: Optional[str] = None
    image_b64: Optional[str] = None


import time

# Simple in-memory cache for recent chat responses
_chat_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached_response(key):
    if key in _chat_cache:
        val, ts = _chat_cache[key]
        if time.time() - ts < _CACHE_TTL:
            return val
        del _chat_cache[key]
    return None


def _set_cache(key, val):
    _chat_cache[key] = (val, time.time())
    # Evict old entries if cache grows too large
    if len(_chat_cache) > 100:
        oldest = min(_chat_cache, key=lambda k: _chat_cache[k][1])
        del _chat_cache[oldest]


def extract_topic_from_query(question: str) -> str:
    """Extract the topic from the user query."""
    q_lower = question.lower()

    # Try common markers
    markers = [
        "problems on",
        "problem on",
        "questions on",
        "question on",
        "pyqs on",
        "pyq on",
        "problems about",
        "questions about",
        "questions for",
        "problems for",
        "find questions on",
        "find problems on",
    ]
    for marker in markers:
        if marker in q_lower:
            idx = q_lower.find(marker)
            extracted = question[idx + len(marker) :].strip("? .!").strip()
            if extracted:
                return extracted.title()

    # Fallback: remove stop words and return the remaining words capitalized
    stop_words_for_topic = {
        "get",
        "me",
        "a",
        "the",
        "all",
        "questions",
        "question",
        "problems",
        "problem",
        "pyqs",
        "pyq",
        "on",
        "about",
        "for",
        "find",
        "show",
        "list",
        "give",
        "related",
        "are",
        "there",
        "any",
        "is",
        "what",
        "how",
        "why",
        "who",
        "where",
        "can",
        "you",
        "tell",
        "explain",
        "describe",
        "provide",
        "please",
        "past",
        "year",
    }
    words = question.split()
    clean_words = []
    for w in words:
        w_clean = w.strip(".,!?-'\"")
        if w_clean.lower() not in stop_words_for_topic:
            clean_words.append(w)

    if clean_words:
        return " ".join(clean_words).title()

    return question.strip("? .!").title()


# ── Conversational Memory ─────────────────────────────────────────
_session_history: dict = defaultdict(list)
_session_last_active: dict = {}
MAX_HISTORY_TURNS = 6  # last 3 user + 3 assistant turns

def get_session_history(session_id: Optional[str]) -> list:
    if not session_id:
        return []
    now = time.time()
    if session_id in _session_last_active:
        if now - _session_last_active[session_id] > 7200:  # 2-hour expiry
            _session_history[session_id] = []
    _session_last_active[session_id] = now
    return list(_session_history[session_id][-MAX_HISTORY_TURNS:])

def append_session_message(session_id: Optional[str], role: str, content: str):
    if not session_id or not content.strip():
        return
    _session_history[session_id].append({"role": role, "content": content.strip()})
    _session_last_active[session_id] = time.time()
    if len(_session_history[session_id]) > MAX_HISTORY_TURNS:
        _session_history[session_id] = _session_history[session_id][-MAX_HISTORY_TURNS:]


@app.post("/chat")
@limiter.limit("15/minute")
async def chat_endpoint(
    request: Request, payload: ChatRequest, db: AsyncSession = Depends(get_db)
):
    """
    Takes a student's question, uses tiered intent routing, searches
    PostgreSQL/Neo4j for relevant context, and streams the AI response back.
    """
    question = payload.message
    session_id = payload.session_id or "default_session"
    append_session_message(session_id, "user", question)
    
    history = get_session_history(session_id)
    cache_key = f"{session_id}:{question.strip().lower()}" if history else question.strip().lower()

    # --- Check cache first ---
    cached = _get_cached_response(cache_key)
    if cached:

        async def yield_cached():
            yield cached

        return StreamingResponse(yield_cached(), media_type="text/plain")

    # LLM query rewriting: expand vague follow-ups using conversation history
    search_query = question
    if history:
        try:
            hist_str = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-4:])
            rewrite_prompt = (
                f"Conversation so far:\n{hist_str}\n\n"
                f"New user message: {question}\n\n"
                "If this message is a vague follow-up that refers to something from the conversation "
                "(e.g. 'that topic', 'unit 2 only', 'same course'), rewrite it as a fully self-contained "
                "academic query. Otherwise return it unchanged. Return ONLY the rewritten query."
            )
            import httpx as _httpx
            async with _httpx.AsyncClient(headers={"ngrok-skip-browser-warning": "true"}) as _c:
                _r = await _c.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": rewrite_prompt,
                          "stream": False, "options": {"temperature": 0.0}},
                    timeout=20.0,
                )
                rewritten = _r.json().get("response", "").strip()
                if rewritten and rewritten.lower() != question.lower():
                    print(f"[ROUTER] Query rewritten: '{question}' -> '{rewritten}'")
                    search_query = rewritten
        except Exception as _e:
            print(f"[ROUTER] Rewrite failed: {_e}")

    context_parts = []

    intent = await classify_query_intent(search_query)
    print(f"[ROUTER] Intent classified as: {intent}")

    neo4j_driver = get_neo4j()

    if intent == "PROBLEM_LIST":
        topic_name = extract_topic_from_query(question)
        import re as _re
        graph_problems = await fetch_problems_by_topic_graph(neo4j_driver, topic_name) if topic_name else []
        keyword_problems = fetch_all_problems_by_topic(neo4j_driver, topic_name) if topic_name else []
        print(f"[ROUTER] Graph RAG: {len(graph_problems)}, Keyword: {len(keyword_problems)} for '{topic_name}'")
        seen_texts, seen_ids, problems = set(), set(), []
        for p in graph_problems + keyword_problems:
            p_id = p.get("id")
            norm = _re.sub(r'\s+', ' ', (p.get("text") or "").strip().lower())
            if norm and norm not in seen_texts and p_id not in seen_ids:
                seen_texts.add(norm)
                if p_id: seen_ids.add(p_id)
                problems.append(p)
        # Add X-Response-Type header so the frontend can detect this is a structured
        # JSON payload (problem card layout) rather than a streaming text response.
        return JSONResponse(
            content={
                "type": "problem_list",
                "topic": topic_name,
                "problems": problems,
            },
            headers={"X-Response-Type": "problem_list"},
        )

    elif intent == "MULTI_HOP_PREREQ":
        records = execute_graph_prereq_query(neo4j_driver, question)
        if records:
            context_parts.append("--- PREREQUISITE GRAPH KNOWLEDGE ---")
            for r in records:
                context_parts.append(
                    f"Course: {r['course']} requires prerequisites: {', '.join(r['all_prerequisites'])}"
                )

    elif intent == "GRAPH_PYQ_MAPPING":
        records = execute_graph_co_query(neo4j_driver, question)
        if records:
            context_parts.append("--- MAPPED QUESTIONS GRAPH KNOWLEDGE ---")
            for r in records:
                context_parts.append(
                    f"Course {r['course_code']} Question (BTL: {r['btl']}, Marks: {r['marks']}): {r['question']}"
                )

    elif intent == "SIMPLE_CURRICULUM":
        syllabus_res = execute_graph_syllabus_query(neo4j_driver, search_query)
        match_type   = syllabus_res.get("match_type")

        if match_type == "exact":
            records = syllabus_res.get("records", [])
            if records:
                c_code = records[0]["course_code"]
                c_name = records[0]["course_name"]
                context_parts.append(f"--- SYLLABUS: {c_code} - {c_name} ---")
                units_dict = {}
                for r in records:
                    u_num   = r.get("unit_number") or 999
                    u_title = r.get("unit_title") or f"Unit {u_num}"
                    if u_num not in units_dict:
                        units_dict[u_num] = {"title": u_title, "topics": list(r.get("topics", []))}
                    else:
                        for t in r.get("topics", []):
                            if t not in units_dict[u_num]["topics"]:
                                units_dict[u_num]["topics"].append(t)
                for u_num in sorted(units_dict):
                    info = units_dict[u_num]
                    context_parts.append(f"  Unit {u_num} - {info['title']}: {', '.join(info['topics'])}")

        elif match_type == "similar_list":
            courses = syllabus_res.get("courses", [])
            context_parts.append("--- MULTIPLE COURSES FOUND ---")
            context_parts.append("Multiple courses match. Please specify which one you mean:")
            for c in courses:
                context_parts.append(f"- {c['course_code']}: {c['course_name']}")

        else:
            # Fallback to hybrid PostgreSQL vector search
            question_embedding = await get_embedding(search_query)
            if question_embedding:
                try:
                    chunks = await hybrid_search_rrf(db, search_query, question_embedding, k=5)
                    if chunks:
                        context_parts.append("--- ADDITIONAL TEXT CHUNKS ---")
                        for chunk in chunks:
                            if chunk.get("content"):
                                context_parts.append(chunk["content"])
                except Exception as e:
                    print(f"Hybrid search error: {e}")

    elif intent == "GENERAL_QUERY":
        # Search PostgreSQL via RRF for extra prose chunks to explain the concept
        question_embedding = await get_embedding(question)
        if question_embedding:
            try:
                chunks = await hybrid_search_rrf(db, question, question_embedding, k=5)
                if chunks:
                    context_parts.append("--- REFERENCE MATERIAL ---")
                    for chunk in chunks:
                        content = chunk.get("content", "")
                        if content:
                            context_parts.append(content)
            except Exception as e:
                print(f"Hybrid search error in /chat: {e}")

    else:
        # SIMPLE_PYQ
        neo4j_driver = get_neo4j()
        neo4j_results = execute_neo4j_pyq_search(neo4j_driver, question)

        if neo4j_results:
            context_parts.append("--- NEO4J PYQ SEARCH RESULTS ---")
            for record in neo4j_results:
                img_url = (
                    record["image_url"].replace(" ", "%20")
                    if record.get("image_url")
                    else None
                )
                img_markdown = (
                    f"\n![Diagram for Q{record['q_num']}]({img_url})"
                    if img_url and img_url != "None"
                    else ""
                )
                context_parts.append(
                    f"[Course: {record['course_code']} - Q: {record['q_num']}]\n{record['q_text']}{img_markdown}\nMarks: {record['marks']}\nBTL: {record['btl']}"
                )
        else:
            question_embedding = await get_embedding(question)
            if question_embedding:
                try:
                    chunks = await hybrid_search_rrf(
                        db, question, question_embedding, k=5
                    )
                    if chunks:
                        context_parts.append(
                            "--- HYBRID SEARCH RESULTS (TEXT + SEMANTIC) ---"
                        )
                        for chunk in chunks:
                            content = chunk.get("content", "")
                            if content:
                                context_parts.append(content)
                except Exception as e:
                    print(f"Hybrid search error in /chat: {e}")

    # Generate final answer
    final_context = "\n".join(context_parts)
    if len(final_context) > 8000:
        final_context = final_context[:8000] + "\n...[Context Truncated]..."

    print("DEBUG FINAL CONTEXT:")
    print(final_context)

    # Stream response and cache it
    async def stream_and_cache():
        full_response = []
        async for chunk in generate_answer_stream(question, final_context):
            full_response.append(chunk)
            yield chunk

        # Append images at the end to guarantee they render, since the LLM often drops them
        import re

        images = re.findall(r"!\[.*?\]\(.*?\)", final_context)
        if images:
            combined_response_str = "".join(full_response)
            unique_images = []
            for img in images:
                # Extract URL from the markdown tag e.g. ![...](url)
                url_match = re.search(r"\((https?://.*?)\)", img)
                if url_match:
                    url = url_match.group(1)
                    if url not in combined_response_str:
                        unique_images.append(img)
                else:
                    if img not in combined_response_str:
                        unique_images.append(img)

            if unique_images:
                image_block = "\n\n### Diagrams from Questions:\n" + "\n\n".join(
                    unique_images
                )
                full_response.append(image_block)
                yield image_block

        response_str = "".join(full_response)
        _set_cache(cache_key, response_str)
        append_session_message(session_id, "assistant", response_str)

    return StreamingResponse(stream_and_cache(), media_type="text/plain")


# Route: /upload — Handles Admin PDF Uploads


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    doc_type: str = Form(...),
    department: str | None = Form(None),
    dept: str | None = Form(
        None
    ),  # new frontend sends 'dept', old sends 'department'
    year: str | None = Form(None),
    course_code: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts single or multiple PDF documents uploaded by the admin.
    Depending on `doc_type` ("syllabus" or "pyq"), dispatches to the correct pipeline.
    """
    os.makedirs("uploads", exist_ok=True)
    # Normalise: accept 'dept' (new frontend) OR 'department' (old frontend / direct API calls)
    department = department or dept

    # Save the files and dispatch tasks
    for file in files:
        file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        new_doc = Document(
            filename=file.filename,
            doc_type=doc_type,
            department=department,
            year=year,
            course_code=course_code,
        )
        db.add(new_doc)
        await db.commit()
        await db.refresh(new_doc)

        if doc_type == "syllabus":
            background_tasks.add_task(
                process_syllabus_background, file_path, department, year, new_doc.id
            )
        elif doc_type == "pyq":
            background_tasks.add_task(
                process_pyq_background, file_path, course_code, new_doc.id
            )

    return {
        "message": f"{len(files)} file(s) uploaded successfully! Processing in the background."
    }


# Route: /documents — Lists All Uploaded Documents


@app.get("/documents")
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    Fetches and returns a list of all documents from the PostgreSQL database,
    ordered by the most recently uploaded first.
    """
    try:
        # Query the Document table, ordering by creation date descending
        result = await db.execute(select(Document).order_by(Document.created_at.desc()))
        docs = result.scalars().all()

        # Format the results into a list of dictionaries
        # doc_type is included so the new frontend can render it in the admin table
        return [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "doc_type": doc.doc_type,
                "course_code": doc.course_code,
                "created_at": doc.created_at.isoformat(),
            }
            for doc in docs
        ]
    except Exception as e:
        print(f"Error fetching document list: {e}")
        return []


# Route: /documents/{id} — Deletes a Specific Document


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """
    Permanently deletes a document and all of its associated text chunks
    from PostgreSQL, Neo4j, and the local file system.
    """
    try:
        # 1. Fetch document metadata first
        doc_uuid = uuid.UUID(doc_id) if isinstance(doc_id, str) else doc_id
        result = await db.execute(select(Document).where(Document.id == doc_uuid))
        doc = result.scalar_one_or_none()

        if doc:
            c_code = doc.course_code.upper() if doc.course_code else None

            # 2. Delete from Neo4j Graph Database
            try:
                neo4j_driver = get_neo4j()
                with neo4j_driver.session() as session:
                    if doc.doc_type == "pyq":
                        # Delete Question nodes tagged with this document_id
                        session.run(
                            """
                            MATCH (q:Question)
                            WHERE q.document_id = $doc_id
                            DETACH DELETE q
                        """,
                            doc_id=str(doc_id),
                        )

                        # Delete orphan QuestionModel nodes
                        session.run("""
                            MATCH (qm:QuestionModel)
                            WHERE NOT (qm)-[:HAS_QUESTION]->()
                            DETACH DELETE qm
                        """)

                        # Delete orphan Course nodes if they have no units and no question models
                        session.run("""
                            MATCH (c:Course)
                            WHERE NOT (c)-[:HAS_UNIT]->() AND NOT (c)-[:HAS_QUESTION_MODEL]->()
                            DETACH DELETE c
                        """)

                    elif doc.doc_type == "syllabus":
                        if c_code:
                            session.run(
                                """
                                MATCH (c:Course {code: $c_code})
                                OPTIONAL MATCH (c)-[:HAS_UNIT]->(u:Unit)
                                OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
                                DETACH DELETE t, u
                            """,
                                c_code=c_code,
                            )
                            session.run(
                                """
                                MATCH (c:Course {code: $c_code})
                                WHERE NOT (c)-[:HAS_UNIT]->() AND NOT (c)-[:HAS_QUESTION_MODEL]->()
                                DETACH DELETE c
                            """,
                                c_code=c_code,
                            )
            except Exception as graph_err:
                print(f"Error cleaning up Neo4j for document {doc_id}: {graph_err}")

            # 3. Clean up physical image files from disk and Cloudinary
            try:
                # Cloudinary delete
                delete_document_images_from_cloudinary(str(doc_id))

                images_dir = "uploads/images"
                if os.path.exists(images_dir):
                    for fname in os.listdir(images_dir):
                        if str(doc_id) in fname:
                            os.remove(os.path.join(images_dir, fname))
            except Exception as fs_err:
                print(f"Error deleting physical files for document {doc_id}: {fs_err}")

        # 4. Delete child chunks and parent document from PostgreSQL
        await db.execute(
            text("DELETE FROM document_chunks WHERE document_id = :id"), {"id": doc_id}
        )
        await db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})

        await db.commit()
        return {
            "message": "Document and its associated graph nodes deleted successfully."
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Route: /reset — Hard Resets the Entire System


@app.post("/reset")
async def reset_system(db: AsyncSession = Depends(get_db)):
    """
    DANGER ZONE: This endpoint completely wipes all data from PostgreSQL, Neo4j,
    and deletes all uploaded files from the disk. Used to start totally fresh.
    """
    errors = []

    # 1. Wipe PostgreSQL (Delete all chunks and documents)
    try:
        await db.execute(text("DELETE FROM document_chunks"))
        await db.execute(text("DELETE FROM documents"))
        await db.commit()
        print("System Reset: PostgreSQL wiped.")
    except Exception as e:
        await db.rollback()
        errors.append(f"PostgreSQL error: {e}")

    # 2. Wipe Neo4j (Delete all nodes and relationships in the graph)
    try:
        neo4j_driver = get_neo4j()
        with neo4j_driver.session() as session:
            # Cypher command to find all nodes and detach/delete them
            session.run("MATCH (n) DETACH DELETE n")
        print("System Reset: Neo4j wiped.")
    except Exception as e:
        errors.append(f"Neo4j error: {e}")

    # 3. Wipe Disk and Cloudinary (Delete local uploads and wipe Cloudinary bucket)
    try:
        delete_all_images_from_cloudinary()

        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)
            os.makedirs(uploads_dir)
        print("System Reset: Uploads folder cleared.")
    except Exception as e:
        errors.append(f"File system / Cloudinary error: {e}")

    # If any step failed, return a 500 error outlining what went wrong
    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    return {"message": "System reset complete. All data has been wiped."}


# Route: /image-query — Student Uploads an Image for Reverse Lookup


@app.post("/image-query")
async def image_query_endpoint(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """
    Accepts an image (photo of an equation, formula, circuit diagram, etc.) from
    a student. The vision model describes the image, and we use that description
    to search both PostgreSQL and Neo4j for related curriculum content.
    """
    # Read the uploaded image and convert it to base64
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Step 1: Ask the vision model to describe what is in the image
    image_description = await describe_uploaded_image(b64_image)

    # Step 2: Use the description as a search query (same hybrid search as /chat)
    context_parts = []

    # --- Vector Search in PostgreSQL ---
    query_embedding = await get_embedding(image_description)
    if query_embedding:
        try:
            query = (
                select(DocumentChunk)
                .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
                .limit(10)
            )
            result = await db.execute(query)
            similar_chunks = result.scalars().all()

            if similar_chunks:
                context_parts.append("--- RELEVANT TEXT FROM DOCUMENTS ---")
                for chunk in similar_chunks:
                    context_parts.append(chunk.content)
        except Exception as e:
            print(f"Image query vector search error: {e}")

    # --- Graph Search in Neo4j ---
    try:
        neo4j_driver = get_neo4j()
        words = image_description.lower().split()

        with neo4j_driver.session() as session:
            # Query the graph: Find any nodes where the name or text matches the student's question words.
            # We search across Topics, SubTopics, QuestionModels, and Questions.
            records = session.run(
                """
                MATCH (n)
                WHERE (n:Topic OR n:SubTopic OR n:QuestionModel OR n:Question)
                AND any(word IN $words WHERE toLower(n.name) CONTAINS word OR toLower(n.text) CONTAINS word OR toLower(n.implicit_formulas) CONTAINS word)
                
                // If it's a question, find its model and course
                OPTIONAL MATCH (c:Course)-[:HAS_QUESTION_MODEL]->(qm:QuestionModel)-[:HAS_QUESTION]->(n:Question)
                
                // If it's a topic, find its course
                OPTIONAL MATCH (c2:Course)-[:HAS_UNIT]->(:Unit)-[:HAS_TOPIC]->(n:Topic)
                
                RETURN labels(n) as labels, n.name as name, n.text as text, n.implicit_formulas as formulas,
                       coalesce(c.code, c2.code) as course_code
                LIMIT 5
                """,
                words=words,
            )

            graph_facts = []
            for r in records:
                labels = r["labels"]
                if "Question" in labels:
                    fact = f"[Course {r['course_code']} - PYQ] {r['text']}"
                    if r["formulas"]:
                        fact += f" (Formulas: {r['formulas']})"
                    graph_facts.append(fact)
                else:
                    graph_facts.append(
                        f"[Course {r['course_code']} - {labels[0]}] {r['name']}"
                    )

            if graph_facts:
                context_parts.append("--- KNOWLEDGE GRAPH FACTS ---")
                context_parts.extend(graph_facts)
    except Exception as e:
        print(f"Image query graph search error: {e}")

    # Step 3: Generate the final answer using all the context
    final_context = "\n".join(context_parts)

    prompt_question = (
        f"A student uploaded an image. Here is what the image contains:\n\n"
        f"{image_description}\n\n"
        f"Based on the curriculum context provided, explain where this concept appears "
        f"in the course materials, what topics it relates to, and what kinds of questions "
        f"or problems typically involve this formula, equation, or diagram."
    )

    ai_response = await generate_answer(prompt_question, final_context)

    return {"description": image_description, "response": ai_response}


# Background Task Logic


async def process_syllabus_background(
    file_path: str, department: str, year: str, document_id: uuid.UUID
):
    import json
    report = {"type": "syllabus", "status": "started", "file": file_path}
    """
    Campus-wide GraphRAG syllabus ingestion using the new SyllabusParser.
    Builds a full dual-layer Neo4j graph (semantic + source/document).
    Supports any department (CSE, ECE, EEE, MECH, ...) and curriculum year.
    """
    from syllabus_parser import SyllabusParser

    print(f"[Syllabus] Starting GraphRAG ingestion for {department} {year}...")
    neo4j_driver = get_neo4j()

    try:
        parser = SyllabusParser(neo4j_driver)
        # parse_and_ingest is synchronous (CPU-bound regex + Neo4j writes);
        # run it in a thread executor to avoid blocking the event loop.
        import asyncio

        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None, parser.parse_and_ingest, file_path, department, year
        )
        print(f"[Syllabus] Ingestion complete: {summary}")
        report["status"] = "success"
        report["summary"] = summary
        async for db in get_db():
            db.add(ProcessingReport(document_id=document_id, course_code="SYLLABUS", report_json=json.dumps(report)))
            await db.commit()
            break
    except Exception as e:
        import traceback

        print(f"[Syllabus] ERROR during ingestion of {department} {year}: {e}")
        report["status"] = "error"
        report["error"] = str(e)
        async for db in get_db():
            db.add(ProcessingReport(document_id=document_id, course_code="SYLLABUS", report_json=json.dumps(report)))
            await db.commit()
            break
        traceback.print_exc()


async def process_pyq_background(
    file_path: str, course_code: str, document_id: uuid.UUID
):
    import json
    report = {"type": "pyq", "course_code": course_code, "status": "started", "structured": 0, "vision": 0, "mapped": 0}
    print(f"[PYQ] Starting processing for {course_code}...")
    neo4j_driver = get_neo4j()

    # Stage 1: Structured Text Extractor (fast, CO/BTL-aware)
    structured_qs = extract_pyq_structured(file_path, course_code, str(document_id))

    if structured_qs:
        print(
            f"[PYQ] Text extractor found {len(structured_qs)} questions. Mapping to KG..."
        )
        await map_pyq_structured_to_kg(neo4j_driver, structured_qs, str(document_id))

        async for db in get_db():
            for q in structured_qs:
                labeled_content = f"[PYQ - {course_code} - {q['question_number']}]\n{q['question_text']}"
                img_url = q.get("image_url")
                if img_url:
                    labeled_content += f"\n\n![Diagram for Q{q['question_number']}]({img_url})"
                embedding = await get_embedding(labeled_content)
                if embedding:
                    text_chunk = DocumentChunk(
                        document_id=document_id,
                        content=labeled_content,
                        course_code=course_code,
                        content_type="text",
                        embedding=embedding,
                    )
                    db.add(text_chunk)
            await db.commit()
            break
        print(f"[PYQ] Finished processing structured text for {course_code}!")
        report["status"] = "success_structured"
        report["structured"] = len(structured_qs)
        async for db in get_db():
            db.add(ProcessingReport(document_id=document_id, course_code=course_code, report_json=json.dumps(report)))
            await db.commit()
            break
        return

    print("[PYQ] Text extractor found 0 questions. Falling back to Vision AI...")

    # Stage 2: Vision pipeline
    # Render pages as chunked images
    page_chunks = process_pyq_visuals(file_path)
    print(f"[PYQ] Rendered {len(page_chunks)} image chunks.")

    async for db in get_db():
        for i, chunk_data in enumerate(page_chunks):
            print(
                f"[PYQ] Extracting questions from chunk {i + 1}/{len(page_chunks)} via Vision AI..."
            )
            try:
                # Extract questions via Vision
                questions = await extract_pyq_questions(chunk_data["base64"])

                if not questions:
                    continue

                # Upload the page image chunk to Cloudinary (or local fallback)
                image_bytes = base64.b64decode(chunk_data["base64"])
                page_num_lbl = chunk_data["page"] + 1
                chunk_index = chunk_data["chunk_index"]
                public_id = f"{document_id}_page_{page_num_lbl}_chunk_{chunk_index}"
                image_url = upload_question_image_to_cloudinary(image_bytes, public_id)

                # Map extracted questions into the KG
                await map_questions_to_kg(
                    neo4j_driver, course_code, questions, document_id, image_url
                )

                for q in questions:
                    if isinstance(q, str):
                        q = {
                            "text": q,
                            "question_number": "Unknown",
                            "likely_topic": "General",
                            "implicit_formulas": [],
                        }

                    q_text = q.get("text", "")

                    # Validate q_text before storing to prevent junk/JSON
                    if len(q_text) < 20 or q_text.startswith("{") or '{"' in q_text:
                        continue
                    if any(
                        junk in q_text[:50].lower()
                        for junk in [
                            "answer all",
                            "part a",
                            "part b",
                            "co |",
                            "course outcomes",
                        ]
                    ):
                        continue

                    # Append the markdown image link so the LLM includes it in the chat
                    markdown_image = f"![PYQ Page - {course_code}]({image_url})"
                    labeled_content = f"[PYQ - {course_code} - {q.get('question_number')}]\n{q_text}\nImplicit Formulas: {', '.join(q.get('implicit_formulas', []))}\n\n{markdown_image}"

                    embedding = await get_embedding(labeled_content)
                    if embedding:
                        visual_chunk = DocumentChunk(
                            document_id=document_id,
                            content=labeled_content,
                            course_code=course_code,
                            content_type="visual",
                            embedding=embedding,
                        )
                        db.add(visual_chunk)
                        await db.commit()

            except Exception as e:
                import traceback

                print(
                    f"[PYQ] Error processing chunk {chunk_data['chunk_index']} on page {chunk_data['page'] + 1}: {e}"
                )
                traceback.print_exc()
                await db.rollback()
        break

    print(f"[PYQ] Finished processing all content for {course_code}!")
    report["status"] = "success_vision"
    async for db in get_db():
        db.add(ProcessingReport(document_id=document_id, course_code=course_code, report_json=json.dumps(report)))
        await db.commit()
        break


# Application Entry Point

# When you run `python app.py`, uvicorn starts the server on port 8000
if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open("http://localhost:8000")

    # Open the browser 1.5 seconds after starting the command,
    # giving Uvicorn enough time to bind to the port.
    Timer(1.5, open_browser).start()

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, access_log=False)
