import base64
import http
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import uvicorn

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
from database import Base, Document, DocumentChunk, engine, get_db, get_neo4j
from query_neo4j import fetch_all_problems_by_topic
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


async def classify_query_intent(question: str) -> str:
    """
    Tiered classification: Fast keyword pre-filter first.
    If ambiguous, invoke the LLM classifier.
    """
    q_lower = question.lower()

    # Tier 1: Fast Keyword Routing
    # Check for problem list requests first
    if any(
        marker in q_lower
        for marker in [
            "problems on",
            "problem on",
            "questions on",
            "question on",
            "pyqs on",
            "pyq on",
            "problems about",
            "questions about",
        ]
    ):
        return "PROBLEM_LIST"

    if any(
        k in q_lower
        for k in [
            "prerequisite",
            "before taking",
            "should i know",
            "requires",
            "needed for",
        ]
    ):
        return "MULTI_HOP_PREREQ"
    elif any(
        k in q_lower
        for k in [
            "btl",
            "co1",
            "co2",
            "co3",
            "co4",
            "co5",
            "bloom",
            "mapped to",
            "course outcome questions",
        ]
    ):
        return "GRAPH_PYQ_MAPPING"
    elif any(
        k in q_lower
        for k in [
            "syllabus",
            "topics",
            "units",
            "course outcomes",
            "objectives",
            "evaluation",
            "pattern",
            "textbook",
            "reference",
        ]
    ):
        return "SIMPLE_CURRICULUM"
    elif any(
        k in q_lower
        for k in [
            "question",
            "questions",
            "pyq",
            "past year",
            "exam",
            "paper",
            "midterm",
            "mid-term",
            "endsem",
            "end-sem",
            "problem",
            "problems",
            "circuit",
        ]
    ):
        return "SIMPLE_PYQ"

    # Tier 2: LLM Fallback (Slow)
    prompt = f"""
You are a query classification engine for a university information system.
Classify the user's question into exactly one primary category.

Categories:
- "PROBLEM_LIST": Questions asking for problem sets, practice problems, or exam questions on a specific academic topic (e.g. "problems on superposition theorem").
- "SIMPLE_CURRICULUM": Questions about courses, syllabus content, units, topics, learning outcomes.
- "SIMPLE_PYQ": Questions asking for a specific, single past year question (e.g., "what is question 5 of course EEE104?").
- "MULTI_HOP_PREREQ": Questions asking about course prerequisites or what to know before taking a course.
- "GRAPH_PYQ_MAPPING": Questions asking for questions mapped to specific Course Outcomes (COs) or Bloom's Taxonomy Levels (BTLs).
- "GENERAL_QUERY": Open-ended concept questions, explanations, or general academic queries (e.g., "what is nodal analysis?", "explain photosynthesis").

Return ONLY the category name. No explanations.
Question: {question}
"""
    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=25.0,
            )
            ans = response.json().get("response", "").strip().upper()
            if ans in [
                "SIMPLE_CURRICULUM",
                "SIMPLE_PYQ",
                "MULTI_HOP_PREREQ",
                "GRAPH_PYQ_MAPPING",
                "PROBLEM_LIST",
                "GENERAL_QUERY",
            ]:
                return ans
    except Exception as e:
        print(f"LLM classification error: {type(e).__name__}: {e}")

    return "SIMPLE_CURRICULUM"  # Default fallback


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


def execute_graph_syllabus_query(neo4j_driver, question: str) -> list:
    """Queries Neo4j for course syllabus structure (Units & Topics)."""
    words = [
        w.strip(".,!?-'\"")
        for w in question.lower().split()
        if len(w) > 2
        and w
        not in [
            "get",
            "me",
            "the",
            "syllabus",
            "for",
            "course",
            "topics",
            "units",
            "what",
            "is",
        ]
    ]
    if not words:
        words = [question.lower()]

    search_query = " ".join(words)

    with neo4j_driver.session() as session:
        # Use the Neo4j full-text index to robustly find the most relevant course
        records = session.run(
            """
            CALL db.index.fulltext.queryNodes("course_name_ft", $q) YIELD node, score
            WITH node LIMIT 1
            
            // Fetch Units
            OPTIONAL MATCH (node)-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
            WITH node, u, collect(DISTINCT t.name) as topics
            WITH node, collect({title: u.title, topics: topics}) as units
            
            // Fetch Evaluation
            OPTIONAL MATCH (node)-[:HAS_EVALUATION_PATTERN]->(ep:EvaluationPattern)
            OPTIONAL MATCH (ep)-[:HAS_INTERNAL]->(int:Internal)-[:HAS_ASSESSMENT]->(i_ac:AssessmentComponent)
            OPTIONAL MATCH (ep)-[:HAS_EXTERNAL]->(ext:External)-[:HAS_ASSESSMENT]->(e_ac:AssessmentComponent)
            WITH node, units, ep, 
                 collect(DISTINCT {name: i_ac.name, marks: i_ac.marks}) as internals,
                 collect(DISTINCT {name: e_ac.name, marks: e_ac.marks}) as externals
                 
            // Fetch Textbooks & References
            OPTIONAL MATCH (node)-[:USES_TEXTBOOK]->(tb:Textbook)
            WITH node, units, ep, internals, externals, collect(DISTINCT tb.text) as textbooks
            OPTIONAL MATCH (node)-[:HAS_REFERENCE]->(ref:Reference)
            
            RETURN 
                node.code as course_code, 
                node.name as course_name, 
                units, 
                ep.ratio as eval_ratio,
                internals,
                externals,
                textbooks,
                collect(DISTINCT ref.text) as references
        """,
            q=search_query,
        ).data()

        # If no units were found (empty course match), fallback to checking topics directly using CONTAINS
        if not records or (len(records) == 1 and not records[0]["units"]):
            topic_records = session.run(
                """
                MATCH (c:Course)-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
                WHERE all(word IN $words WHERE toLower(t.name) CONTAINS word)
                RETURN c.code as course_code, c.name as course_name, u.title as unit_title, collect(DISTINCT t.name) as topics
                LIMIT 15
            """,
                words=words,
            ).data()
            if topic_records:
                records = topic_records

    return records


class ChatRequest(BaseModel):
    """Defines the expected JSON structure when a student asks a question."""

    message: str
    session_id: str | None = (
        None  # Client-side session tracking; unused server-side for now
    )


import time

_pyq_semaphore = asyncio.Semaphore(1)  # only 1 PYQ task at a time


from collections import defaultdict

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
    if len(_chat_cache) > 100:
        oldest = min(_chat_cache, key=lambda k: _chat_cache[k][1])
        del _chat_cache[oldest]


def extract_topic_from_query(question: str, history: list = None) -> str:
    """Extract the topic from the user query, resolving pronouns/references from history."""
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
                if any(ref in extracted.lower() for ref in referential_phrases) and history:
                    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                    if last_user_msg:
                        return extract_topic_from_query(last_user_msg, None)
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
    cache_key = question.strip().lower()

    # --- Check cache first ---
    cached = _get_cached_response(cache_key)
    if cached:

        async def yield_cached():
            yield cached

        return StreamingResponse(yield_cached(), media_type="text/plain")

    context_parts = []

    intent = await classify_query_intent(question)
    print(f"[ROUTER] Intent classified as: {intent}")

    neo4j_driver = get_neo4j()

    if intent == "PROBLEM_LIST":
        topic_name = extract_topic_from_query(question)
        problems = fetch_all_problems_by_topic(neo4j_driver, topic_name)
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
        records = execute_graph_co_query(neo4j_driver, search_query)
        if records:
            context_parts.append("--- MAPPED QUESTIONS GRAPH KNOWLEDGE ---")
            for r in records:
                context_parts.append(
                    f"Course {r['course_code']} Question (BTL: {r['btl']}, Marks: {r['marks']}): {r['question']}"
                )

    elif intent == "SIMPLE_CURRICULUM":
        # Search Neo4j Graph for Comprehensive Course Profile (Syllabus, Evaluation, etc)
        records = execute_graph_syllabus_query(neo4j_driver, question)
        if records:
            # We get at most 1 comprehensive record or up to 15 simple topic records from fallback
            # Since the comprehensive query aggregates into lists, we can process directly
            for r in records:
                c_code = r.get("course_code")
                c_name = r.get("course_name")
                fact = f"Course Profile for [{c_code}] {c_name}:\n"

                # Check if it's the new comprehensive format or fallback format
                if "units" in r and isinstance(r["units"], list):
                    # Syllabus
                    if r["units"] and r["units"][0].get("title"):
                        fact += "\nSyllabus:\n"
                        for u in r["units"]:
                            if u.get("title"):
                                fact += (
                                    f"  - {u['title']}: "
                                    + ", ".join(u["topics"])
                                    + "\n"
                                )

                    # Evaluation Pattern
                    if r.get("eval_ratio"):
                        fact += f"\nEvaluation Pattern ({r['eval_ratio']}):\n"
                        for i_ac in r.get("internals", []):
                            if i_ac.get("name"):
                                fact += f"  - Internal: {i_ac['name']} ({i_ac['marks']} marks)\n"
                        for e_ac in r.get("externals", []):
                            if e_ac.get("name"):
                                fact += f"  - External: {e_ac['name']} ({e_ac['marks']} marks)\n"

                    # Textbooks
                    if r.get("textbooks"):
                        fact += "\nTextbooks:\n"
                        for tb in r["textbooks"]:
                            if tb:
                                fact += f"  - {tb}\n"

                    # References
                    if r.get("references"):
                        fact += "\nReferences:\n"
                        for ref in r["references"]:
                            if ref:
                                fact += f"  - {ref}\n"

                    context_parts.append(fact)
                else:
                    # Fallback topic logic
                    if r.get("unit_title") and r.get("topics"):
                        fact += (
                            f"  - {r['unit_title']}: " + ", ".join(r["topics"]) + "\n"
                        )
                        context_parts.append(fact)

        match_type = syllabus_res.get("match_type")
        if match_type == "exact":
            records = syllabus_res.get("records", [])
            c_code = records[0]["course_code"]
            c_name = records[0]["course_name"]
            context_parts.append(f"--- EXACT MATCHED SYLLABUS: {c_code} - {c_name} ---")
            context_parts.append(f"Course: {c_code} - {c_name}")

            # Group topics by unit number and sort strictly ascending
            units_dict = {}
            for r in records:
                u_num = r.get("unit_number") or 999
                u_title = r.get("unit_title") or f"Unit {u_num}"
                topics = r.get("topics", [])
                if u_num not in units_dict:
                    units_dict[u_num] = {"title": u_title, "topics": list(topics)}
                else:
                    for t in topics:
                        if t not in units_dict[u_num]["topics"]:
                            units_dict[u_num]["topics"].append(t)

            for u_num in sorted(units_dict.keys()):
                u_info = units_dict[u_num]
                topics_str = ", ".join(u_info["topics"])
                context_parts.append(f"  Unit {u_num}: {topics_str}")

        elif match_type == "similar_list":
            courses = syllabus_res.get("courses", [])
            context_parts.append("--- MULTIPLE SIMILAR COURSES FOUND (NO EXACT MATCH) ---")
            context_parts.append("No single exact match was found. The curriculum contains these related courses:")
            for c in courses:
                context_parts.append(f"- {c['course_code']}: {c['course_name']}")
            context_parts.append("List these matching courses for the student and ask which one they want the syllabus for.")

        else:
            # ── Fallback: only if Neo4j returned nothing ─────────────────────
            question_embedding = await get_embedding(search_query)
            if question_embedding:
                try:
                    chunks = await hybrid_search_rrf(db, search_query, question_embedding, k=5)
                    if chunks:
                        context_parts.append("--- ADDITIONAL TEXT CHUNKS ---")
                        for chunk in chunks:
                            content = chunk.get("content", "")
                            if content:
                                context_parts.append(content)
                except Exception as e:
                    print(f"Hybrid search error in /chat: {e}")

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
        # SIMPLE_PYQ / GENERAL
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
            question_embedding = await get_embedding(search_query)
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

    # --- Context Re-ranking ---
    try:
        from embedder import embed_text, cosine_similarity
        q_emb = await embed_text(search_query)
        if q_emb and len(context_parts) > 1:
            scored_parts = []
            for part in context_parts:
                if part.startswith("---"):
                    # keep headers high but don't re-rank them strictly
                    scored_parts.append((1.0, part))
                    continue
                p_emb = await embed_text(part[:1000]) # embed start of part
                if p_emb:
                    sim = cosine_similarity(q_emb, p_emb)
                    scored_parts.append((sim, part))
                else:
                    scored_parts.append((0.0, part))
            # Sort by descending similarity
            scored_parts.sort(key=lambda x: x[0], reverse=True)
            context_parts = [p for _, p in scored_parts]
    except Exception as e:
        print(f"Context re-ranking error: {e}")

    # Generate final answer
    final_context = "\n".join(context_parts)
    if len(final_context) > 8000:
        final_context = final_context[:8000] + "\n...[Context Truncated]..."

    print("DEBUG FINAL CONTEXT:")
    print(final_context)

    # Stream response and cache it
    async def stream_and_cache():
        full_response = []
        async for chunk in generate_answer_stream(question, final_context, history=history):
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

        _set_cache(cache_key, "".join(full_response))

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
    except Exception as e:
        import traceback

        print(f"[Syllabus] ERROR during ingestion of {department} {year}: {e}")
        traceback.print_exc()


async def process_pyq_background(
    file_path: str, course_code: str, document_id: uuid.UUID
):
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
                labeled_content = (
                    f"[PYQ - {course_code} - {q['question_number']}]\n{q['question_text']}"
                )
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
                    embed_count += 1
            await db.commit()
            break
        report.embedded_postgres = embed_count
        report.timings["embed_s"] = round(time.time() - t_embed_start, 2)

        # ── Stage 4: Topic mapping (async, best-effort) ─────────────────────
        try:
            t_map_start = time.time()
            map_result = await topic_mapper_run_for_document(
                neo4j_driver, doc_id_str, report
            )
            report.mapped_to_topic = map_result.get("mapped", 0)
            report.timings["topic_map_s"] = round(time.time() - t_map_start, 2)
        except Exception as map_err:
            print(f"[PYQ] Topic mapping error (non-fatal): {map_err}")

        report.finalize()
        final_status = report.status
        print(f"[PYQ] Finished processing structured text for {course_code}!")
        return

    print("[PYQ] Text extractor found 0 questions. Falling back to Vision AI...")

    else:
        # ── Stage 1b: Vision pipeline fallback (scanned PDFs) ──────────────
        print(f"[PYQ] Text extractor found 0 questions. Falling back to Vision AI...")
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
                        embedding = await get_embedding(labeled_content)
                        if embedding:
                            visual_chunk = DocumentChunk(
                                document_id=document_id,
                                content=labeled_content,
                                course_code=course_code,
                                content_type="visual",
                                embedding=embedding
                            )
                            db.add(visual_chunk)
                            report.saved_neo4j += 1
                            report.embedded_postgres += 1
                            await db.commit()

            except Exception as e:
                import traceback

                print(
                    f"[PYQ] Error processing chunk {chunk_data['chunk_index']} on page {chunk_data['page'] + 1}: {e}"
                )
                traceback.print_exc()
                await db.rollback()
        break

        # ── Stage 4: Topic mapping (async, best-effort) for Vision Fallback ──
        try:
            t_map_start = time.time()
            map_result = await topic_mapper_run_for_document(
                neo4j_driver, doc_id_str, report
            )
            report.mapped_to_topic = map_result.get("mapped", 0)
            report.timings["topic_map_s"] = round(time.time() - t_map_start, 2)
        except Exception as map_err:
            print(f"[PYQ] Topic mapping error (non-fatal): {map_err}")

        report.finalize()
        final_status = report.status
        print(f"[PYQ] Finished vision fallback for {course_code}!")

    # ── Persist ProcessingReport to PostgreSQL ──────────────────────────────
    try:
        async for db in get_db():
            pr = ProcessingReport(
                document_id=document_id,
                course_code=course_code,
                report_json=report.to_json(),
            )
            db.add(pr)
            await db.execute(
                text("UPDATE documents SET processing_status = :status WHERE id = :id"),
                {"status": final_status, "id": doc_id_str}
            )
            await db.commit()
            break
    except Exception as save_err:
        print(f"[PYQ] Failed to save ProcessingReport: {save_err}")
        async for db in get_db():
            await db.execute(
                text("UPDATE documents SET processing_status = :status WHERE id = :id"),
                {"status": final_status, "id": doc_id_str}
            )
            await db.commit()
            break

    print(f"[PYQ] Processing complete for {course_code} — status: {final_status}.")


# =============================================================================
# Admin Routes — Extraction Review
# =============================================================================

def _require_admin(authorization: str = None):
    """Verify the request carries a valid admin JWT. Returns the payload."""
    from fastapi import Header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return payload


@app.get("/admin/extraction-review/courses")
async def admin_list_courses(authorization: Optional[str] = Header(None)):
    """List all courses with their extraction status and topic counts."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (c:Course)-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
            RETURN c.code AS code, c.name AS name,
                   count(DISTINCT u) AS unit_count,
                   count(DISTINCT t) AS topic_count,
                   count(DISTINCT CASE WHEN t.approved = true THEN t END) AS approved_count,
                   count(DISTINCT CASE WHEN t.approved = false THEN t END) AS pending_count
            ORDER BY c.code
        """).data()
    return {"courses": records}


@app.get("/admin/extraction-review/courses/{course_code}")
async def admin_course_review(course_code: str, authorization: Optional[str] = Header(None)):
    """Return units + topics for a course, including unit raw_text for admin review."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (c:Course {code: $code})-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
            RETURN u.id AS unit_id, u.number AS unit_number, u.title AS unit_title,
                   u.raw_text AS unit_raw_text,
                   collect({
                       id: t.id, name: t.name, approved: t.approved,
                       extraction_method: t.extraction_method,
                       extraction_confidence: t.extraction_confidence
                   }) AS topics
            ORDER BY u.number
        """, code=course_code).data()
    return {"course_code": course_code, "units": records}


class TopicUpdateRequest(BaseModel):
    name: Optional[str] = None
    approved: Optional[bool] = None

@app.put("/admin/extraction-review/topics/{topic_id}")
async def admin_update_topic(
    topic_id: str, body: TopicUpdateRequest, authorization: Optional[str] = None
):
    """Edit or approve/reject a Topic node."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
        updates["normalized_name"] = body.name.lower().strip()
    if body.approved is not None:
        updates["approved"] = body.approved
    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")
    set_clause = ", ".join(f"t.{k} = ${k}" for k in updates)
    with neo4j_driver.session() as session:
        session.run(f"MATCH (t:Topic {{id: $id}}) SET {set_clause}",
                    id=topic_id, **updates)
    return {"status": "updated", "topic_id": topic_id}


class TopicMergeRequest(BaseModel):
    source_id: str
    target_id: str

@app.post("/admin/extraction-review/topics/merge")
async def admin_merge_topics(body: TopicMergeRequest, authorization: Optional[str] = None):
    """Merge source topic into target: move all relationships then delete source."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        # Move HAS_TOPIC relationships
        session.run("""
            MATCH (u:Unit)-[:HAS_TOPIC]->(src:Topic {id: $src_id})
            MATCH (tgt:Topic {id: $tgt_id})
            MERGE (u)-[:HAS_TOPIC]->(tgt)
        """, src_id=body.source_id, tgt_id=body.target_id)
        # Move TESTS_TOPIC relationships
        session.run("""
            MATCH (q:Question)-[r:TESTS_TOPIC]->(src:Topic {id: $src_id})
            MATCH (tgt:Topic {id: $tgt_id})
            MERGE (q)-[:TESTS_TOPIC {confidence: r.confidence, review_status: r.review_status,
                                     mapping_method: r.mapping_method, mapped_at: r.mapped_at}]->(tgt)
        """, src_id=body.source_id, tgt_id=body.target_id)
        # Delete source
        session.run("MATCH (t:Topic {id: $id}) DETACH DELETE t", id=body.source_id)
    return {"status": "merged", "source_id": body.source_id, "target_id": body.target_id}


@app.post("/admin/extraction-review/courses/{course_code}/approve")
async def admin_approve_course_topics(
    course_code: str, authorization: Optional[str] = Header(None)
):
    """Mark all Topics for this course as approved."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (c:Course {code: $code})-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
            SET t.approved = true
            RETURN count(t) AS approved_count
        """, code=course_code).single()
    count = result["approved_count"] if result else 0
    return {"status": "approved", "course_code": course_code, "topics_approved": count}


# =============================================================================
# Admin Routes — PYQ Processing Status
# =============================================================================

@app.get("/admin/pyq-processing/{document_id}")
async def admin_pyq_processing_status(
    document_id: str, authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Return the full ProcessingReport for a PYQ document."""
    _require_admin(authorization)
    result = await db.execute(
        text("SELECT report_json FROM processing_reports WHERE document_id = :id ORDER BY created_at DESC LIMIT 1"),
        {"id": document_id}
    )
    row = result.first()
    if not row:
        # Try to get at least the document status
        doc_result = await db.execute(
            text("SELECT processing_status, filename, course_code FROM documents WHERE id = :id"),
            {"id": document_id}
        )
        doc = doc_result.first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {
            "document_id": document_id,
            "processing_status": doc[0],
            "filename": doc[1],
            "course_code": doc[2],
            "report": None
        }
    import json as _json
    report = _json.loads(row[0])
    return {"document_id": document_id, "report": report}


@app.get("/admin/diagnostics/pyq/{course_code}")
async def admin_pyq_diagnostics_summary(
    course_code: str, authorization: Optional[str] = Header(None)
):
    """Return PYQ stats for a course using count(DISTINCT q) to avoid double-counting."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (q:Question)-[:BELONGS_TO]->(c:Course {code: $code})
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[tr:TESTS_TOPIC]->(:Topic)
            RETURN
                count(DISTINCT q) AS total_questions,
                count(DISTINCT CASE WHEN co IS NOT NULL THEN q END) AS questions_with_co,
                count(DISTINCT CASE WHEN tr IS NOT NULL AND tr.review_status = 'approved' THEN q END) AS questions_mapped_to_topic,
                count(DISTINCT q.document_id) AS source_documents
        """, code=course_code).single()

        per_doc = session.run("""
            MATCH (q:Question)-[:BELONGS_TO]->(c:Course {code: $code})
            RETURN q.document_id AS document_id, count(DISTINCT q) AS question_count
            ORDER BY question_count DESC
        """, code=course_code).data()

    return {
        "course_code": course_code,
        "summary": dict(result) if result else {},
        "per_document": per_doc
    }


@app.get("/admin/diagnostics/pyq/{course_code}/questions")
async def admin_pyq_questions_detail(
    course_code: str, authorization: Optional[str] = Header(None)
):
    """Return all questions for a course with mapping details."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (q:Question)-[:BELONGS_TO]->(c:Course {code: $code})
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[tr:TESTS_TOPIC]->(t:Topic)
            WITH q, co, collect(t.name) AS topics, collect(tr.confidence) AS confidences
            RETURN DISTINCT
                q.document_id       AS upload_id,
                q.question_number   AS question_number,
                q.text              AS question,
                q.marks             AS marks,
                q.btl               AS btl,
                co.id               AS course_outcome,
                topics,
                confidences,
                q.image_url         AS image_url
            ORDER BY upload_id, toInteger(q.question_number)
        """, code=course_code).data()
        
        for record in records:
            if record.get("topics"):
                valid_topics = [t for t in record["topics"] if t]
                record["mapped_syllabus_topic"] = ", ".join(valid_topics) if valid_topics else None
            else:
                record["mapped_syllabus_topic"] = None
                
            if record.get("confidences"):
                valid_conf = [c for c in record["confidences"] if c]
                record["mapping_confidence"] = valid_conf[0] if valid_conf else None
            else:
                record["mapping_confidence"] = None

    return {"course_code": course_code, "questions": records, "total": len(records)}


# =============================================================================
# Admin Routes — Topic Mapping Review
# =============================================================================

@app.get("/admin/topic-mapping-review/{course_code}")
async def admin_topic_mapping_review(
    course_code: str, authorization: Optional[str] = Header(None)
):
    """List all TESTS_TOPIC relationships for a course, grouped by review_status."""
    _require_admin(authorization)
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        records = session.run("""
            MATCH (q:Question)-[r:TESTS_TOPIC]->(t:Topic)
            WHERE q.course_code = $code
            RETURN
                id(r)               AS relationship_id,
                q.id                AS question_id,
                q.question_number   AS question_number,
                q.text              AS question_text,
                t.id                AS topic_id,
                t.name              AS topic_name,
                r.confidence        AS confidence,
                r.semantic_score    AS semantic_score,
                r.keyword_score     AS keyword_score,
                r.review_status     AS review_status,
                r.mapped_at         AS mapped_at
            ORDER BY r.review_status, r.confidence DESC
        """, code=course_code).data()
    return {"course_code": course_code, "mappings": records, "total": len(records)}


class MappingUpdateRequest(BaseModel):
    review_status: str   # "approved" | "rejected"
    topic_id: Optional[str] = None   # for re-assignment

@app.put("/admin/topic-mapping-review/{relationship_id}")
async def admin_update_mapping(
    relationship_id: int, body: MappingUpdateRequest,
    authorization: Optional[str] = Header(None)
):
    """Approve, reject, or re-assign a TESTS_TOPIC relationship."""
    _require_admin(authorization)
    if body.review_status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="review_status must be 'approved' or 'rejected'.")
    neo4j_driver = get_neo4j()
    with neo4j_driver.session() as session:
        session.run("""
            MATCH ()-[r:TESTS_TOPIC]->() WHERE id(r) = $rid
            SET r.review_status = $status
        """, rid=relationship_id, status=body.review_status)
    return {"status": "updated", "relationship_id": relationship_id,
            "review_status": body.review_status}


# =============================================================================
# Route: /graph/{course_code} — Bidirectional course graph visualisation
# =============================================================================

@app.get("/graph/{course_code}")
async def get_course_graph(course_code: str, limit: int = 200):
    """
    Returns nodes and edges for the full course graph including BOTH:
    - Outgoing: Course → Unit → Topic (syllabus structure)
    - Incoming: Question → BELONGS_TO → Course (canonical questions)
    - Question → TESTS_TOPIC → Topic (Graph RAG edges)
    """
    neo4j_driver = get_neo4j()
    graph = fetch_graph_for_course(neo4j_driver, course_code.upper(), limit=limit)
    return graph


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

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, access_log=False)
