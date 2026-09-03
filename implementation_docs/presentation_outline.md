M.A.C.H. — Panel Review 1 | Slide Content Reference
23CSE498 — Project Phase 2 | Type 3: Software Project | Team 81
=============================================================

TITLE SLIDE
===========
Project Title : M.A.C.H. — Multimodel Academic Cognitive Hub
               A Fine-tuned Graph-RAG Framework for Campus Knowledge Systems
Course        : 23CSE498 — Project Phase 2 (Panel Review 1)
Team          : 81

  1. CB.SC.U4CSE23505 — A. Chandra Vamsi
  2. CB.SC.U4CSE23508 — B. Prem Sai Charan
  3. CB.SC.U4CSE23530 — M. Pavan Prem Prabhas
  4. CB.SC.U4CSE23568 — P. Tirupati Reddy

Guide : Dr. Shanmuga Priya S
        Assistant Professor (Sl.Gd.), Department of Computer Science & Engineering
        Amrita Vishwa Vidyapeetham, Coimbatore


=============================================================
SLIDE 1 — Software Architecture & Module Design  [5 Marks]
=============================================================
Rubric covers: UML Diagrams, DB Schema, API specs, Module Interaction, Deployment


--- Modular Decomposition & Contracts ---

Auth Module
  The authentication system uses JWT (JSON Web Tokens). When a user submits their
  credentials, the /login endpoint validates them against the PostgreSQL database.
  On success, a signed JWT is issued containing the user's role (student or admin).
  Every protected route verifies this token using FastAPI's Depends() injection.
  Role-based access ensures students cannot reach admin-only endpoints.

Data Layer
  Two separate databases work in parallel.
  PostgreSQL with the pgvector extension stores document text chunks alongside
  their vector embeddings (768-dimensional, produced by the nomic-embed-text model).
  Neo4j stores the curriculum knowledge graph — topics, subjects, and departments
  as nodes, with edges representing prerequisite relationships and curriculum
  hierarchies.

Business Engine
  The Ollama LLM (gemma4:12b-it-qat) runs locally for privacy and zero API cost.
  The RAG pipeline retrieves the top-k semantically similar chunks from pgvector,
  enriches them with graph context from Neo4j, then constructs a structured prompt
  for Ollama to generate the final answer.
  The Syllabus Parser uses Vision AI to process PDF pages as images, extracting
  structured topic and prerequisite data to build the Neo4j graph automatically.
  Cloudinary handles image storage for extracted PDF diagrams, with a local
  fallback when credentials are absent.

API Gateway
  FastAPI serves as the sole entry point at http://0.0.0.0:8000.
  Static frontend files (HTML, CSS, JS) are mounted at /public so the entire
  application is served from one process with no separate web server needed.


--- Core API Endpoints ---

POST /login
  Accepts username and password as form data (application/x-www-form-urlencoded).
  Returns a signed JWT access token. Used by all three pages (login, student, admin).

POST /ask
  The primary RAG endpoint. Accepts a question string and optional filters.
  The response is streamed back to the browser using Server-Sent Events (SSE),
  meaning each token from Ollama appears on screen as it is generated — no waiting.

POST /upload
  Accepts a PDF file upload from the admin dashboard.
  Triggers the full ingestion pipeline: extract text, split into chunks, embed
  each chunk with nomic-embed-text, store vectors in PostgreSQL/pgvector.
  A SHA-256 hash check prevents re-indexing duplicate files.

POST /ingest-syllabus
  Triggers the Syllabus Parser on a specified document.
  Converts PDF pages to images, sends them to Vision AI for structured extraction,
  then writes Topic, Subject, Department, and Year nodes into Neo4j along with
  HAS_PREREQUISITE and BELONGS_TO edges.

GET /graph/prerequisites
  Returns the full prerequisite graph data as JSON.
  Used by the admin dashboard's interactive graph visualization panel.
  Queries Neo4j directly via the Bolt protocol.

GET /documents
  Returns the list of all indexed documents stored in the PostgreSQL documents table.
  Used by the admin dashboard to populate the document management table.


--- Database Schema ---

PostgreSQL — documents table
  Stores metadata for every uploaded PDF.
  Columns: id (UUID), filename (text), department (text), year (int), upload_date (timestamp).

PostgreSQL — document_chunks table
  Stores the actual chunked content and its embedding.
  Columns: id (UUID), document_id (foreign key), content (text), embedding (vector(768)).
  An ivfflat index on the embedding column enables fast approximate nearest-neighbour
  cosine similarity search across thousands of chunks in milliseconds.

Neo4j Knowledge Graph
  Nodes  : Topic, Subject, Department, Year
  Edges  : HAS_PREREQUISITE (Topic → Topic)
           COVERED_IN       (Topic → Subject)
           BELONGS_TO       (Subject → Department)
  Allows the system to answer questions like "What do I need to know before learning
  Operating Systems?" by traversing the graph rather than relying only on keyword
  matching.


--- Deployment Architecture ---

The system is fully containerised using Docker and orchestrated with docker-compose.

  Container 1 — PostgreSQL (with pgvector extension pre-installed)
    Port 5432 exposed internally. Stores documents, chunks, and embeddings.

  Container 2 — Neo4j
    Bolt protocol on port 7687, HTTP browser on port 7474.
    Stores the curriculum knowledge graph.

  Container 3 — FastAPI Backend
    Port 8000 exposed to the host.
    Serves both the REST API and the static frontend from a single process.

  Ollama runs locally on the host machine (not containerised) at port 11434.
  The FastAPI container communicates with Ollama via the host network.

  Environment configuration is managed entirely through a .env file and read at
  startup by the pydantic_settings Settings class, so no secrets are hardcoded.


[INSERT: System Architecture Diagram here]
  Show: Browser → FastAPI (port 8000) → PostgreSQL + Neo4j + Ollama + Cloudinary


=============================================================
SLIDE 2 — Implementation Progress (~60%)  [8 Marks]
=============================================================
Rubric covers: Backend, Frontend, Database & Core Modules Functional


--- Implementation Status ---

Frontend UI — FUNCTIONAL
  Three fully styled, responsive HTML pages:
  Login page (index.html) — Features a live Canvas-based Knowledge Graph animation
  on the left panel, showing the RAG pipeline visually in motion. The animation
  simulates a query flowing through graph nodes in real time.
  Student page (student.html) — The main chat interface. Supports streaming
  responses (tokens appear as Ollama generates them), Markdown rendering via
  marked.js, math rendering via KaTeX, and XSS protection via DOMPurify.
  Admin page (admin.html) — Dashboard with document upload, indexed document list,
  prerequisite graph visualisation (via Chart.js), and system health indicators.
  The entire frontend is PWA-ready with a manifest.json and service-worker.js for
  offline caching and mobile home-screen installation.

Backend Server & APIs — FUNCTIONAL
  FastAPI with Uvicorn serves all routes from a single main.py entry point.
  JWT authentication with 8-hour token expiry protects all non-public routes.
  SlowAPI rate limiting is applied per IP address to prevent abuse.
  The /ask endpoint uses Python's async generator to stream Ollama's response
  token-by-token over SSE — the client sees output appear in real time.

Database Layer — INTEGRATED
  PostgreSQL tables are created automatically on server startup using SQLAlchemy's
  Base.metadata.create_all(), so no manual migration is needed during development.
  The pgvector extension is verified at startup.
  Neo4j connection is established at startup, indexes are created if missing, and
  the prerequisite graph is seeded from the PREREQUISITE_MAP dictionary on every
  startup to ensure consistency.

RAG Pipeline — FUNCTIONAL
  Step 1 — PDF text is extracted using PyMuPDF.
  Step 2 — Text is split into overlapping chunks (512 tokens, 64 token overlap)
  to preserve context across chunk boundaries.
  Step 3 — Each chunk is embedded using the nomic-embed-text model via Ollama.
  Step 4 — Embeddings are stored in PostgreSQL with pgvector.
  Step 5 — At query time, the question is embedded, and the top-k most similar
  chunks are retrieved using cosine similarity.
  Step 6 — The classify_query_intent() function decides whether to augment the
  retrieved chunks with Neo4j graph context (for curriculum/prerequisite questions)
  or use them as-is (for factual/concept questions).
  Step 7 — A structured prompt is built and sent to Ollama, whose response is
  streamed back to the student browser.

Knowledge Graph — FUNCTIONAL
  The Syllabus Parser converts each PDF page to an image at 200 DPI.
  A Vision AI model (via Cloudinary or local fallback) analyses the image and
  returns a structured JSON of topics, subtopics, and prerequisite mappings.
  The SyllabusParser class then creates Neo4j nodes and edges from this data.
  When a student asks a question involving prerequisites or subject relationships,
  the RAG pipeline queries Neo4j first to gather graph context, which is prepended
  to the retrieved text chunks before being sent to the LLM.

Vision AI & Cloudinary — INTEGRATED
  Cloudinary is used for cloud storage of PDF page images during Vision AI
  processing. If Cloudinary credentials are absent in .env, the system silently
  falls back to local disk storage — no crash, no error for the user.
  The cloudinary.uploader.upload() and cloudinary.api.delete_resources() calls
  are fully integrated into the upload and delete lifecycle of documents.

System Integration — VERIFIED
  Full end-to-end flow tested and working:
  Login → JWT issued → Student chat page loads → Question submitted →
  RAG pipeline runs → Streaming response appears in browser →
  Admin uploads PDF → Chunks stored in PostgreSQL → Graph nodes written to Neo4j →
  Student can immediately query the new content.


[INSERT: Screenshot — Student Chat page showing streaming response]
[INSERT: Screenshot — Admin Dashboard showing document list and upload panel]


=============================================================
SLIDE 3a — Technical Knowledge: Architectural Patterns  [5 Marks]
=============================================================
Rubric covers: Software Architecture, Implementation choices, Tech Stack rationale


--- Layered Architecture ---

The system is built on a strict four-layer architecture where each layer only
communicates with the layer directly below it.

  Layer 1 — Presentation
    HTML, CSS, JavaScript. Three pages (index, student, admin).
    The frontend only speaks to the backend via HTTP/SSE — it has no direct
    database access whatsoever.

  Layer 2 — API (FastAPI)
    Handles routing, authentication, rate limiting, and request validation.
    All business logic is triggered from here but not implemented here.

  Layer 3 — Business Logic
    The RAG pipeline (in utils.py), the Syllabus Parser (syllabus_parser.py),
    query intent classification, graph context injection. This layer is
    completely decoupled from the HTTP layer above it.

  Layer 4 — Data
    PostgreSQL (structured data + vectors) and Neo4j (graph data).
    Neither the API layer nor the presentation layer ever writes SQL or Cypher
    directly — they go through the ORM and the driver respectively.

This separation makes each component independently testable and replaceable.


--- Graph-RAG Pattern (Novel Contribution) ---

Standard RAG retrieves text chunks by vector similarity alone.
Our system adds a second retrieval layer using the Neo4j knowledge graph.

  Standard RAG path (used for factual questions):
    User question → embed → pgvector cosine search → top-5 chunks → LLM prompt

  Graph-RAG path (used for curriculum/prerequisite questions):
    User question → classify_query_intent() detects curriculum intent →
    Neo4j traversal to find prerequisite topics and related subjects →
    Results merged with pgvector chunks → enriched LLM prompt →
    Answer includes curriculum context the vector store alone cannot provide.

This dual-layer approach is what differentiates M.A.C.H. from a simple
document chatbot — it understands how subjects relate to each other.


--- Monolith-First, Modular-Ready ---

The current architecture uses a single main.py intentionally during the
research and prototyping phase. This reduces overhead and makes it easy to
iterate quickly on the RAG pipeline and graph queries.

The codebase is already structured with clean boundaries:
  config.py     — all configuration (JWT, Ollama, database settings)
  database.py   — all SQLAlchemy models and session management
  utils.py      — all RAG pipeline, embedding, and graph helper functions
  syllabus_parser.py — the Vision AI PDF processing pipeline

The planned next step (documented in enhancements.md) is to split main.py into
FastAPI APIRouter modules: auth.py, chat.py, upload.py, system.py.
This transition will require zero changes to the frontend or the database.


[INSERT: RAG Pipeline Flowchart]
  Query → Intent Classify → [Vector Search] + [Neo4j Traversal] → Merge Context → LLM → Stream


=============================================================
SLIDE 3b — Technical Knowledge: Patterns, Integrity & Testing  [5 Marks]
=============================================================


--- Design Patterns & Code Structure ---

Dependency Injection
  FastAPI's Depends() system is used throughout. The get_db() function yields
  a PostgreSQL session that is automatically closed after each request, even
  if an exception occurs. The JWT verification function is also injected as a
  dependency — adding authentication to any route requires only one line.

Repository Pattern
  database.py contains all SQLAlchemy ORM model definitions (Document,
  DocumentChunk) and the session factory. Business logic in main.py and utils.py
  never writes raw SQL — it goes through the ORM, keeping data access logic in
  one place and making future database swaps straightforward.

Configuration Singleton
  pydantic_settings' BaseSettings class reads the .env file exactly once at
  process startup. The resulting settings object is a module-level singleton
  imported wherever configuration values are needed. No hardcoded secrets exist
  anywhere in the codebase.

Strategy Pattern
  The classify_query_intent() function implements the Strategy pattern.
  It analyses the user's question and returns a retrieval strategy label
  (e.g. "curriculum", "factual", "comparison"). The RAG pipeline then selects
  the appropriate retrieval path — vector-only or graph-augmented — based on
  this label. New retrieval strategies can be added without modifying existing
  pipeline code.


--- Data Integrity & Concurrency ---

ACID Transactions
  Every write to PostgreSQL is wrapped in a SQLAlchemy session with explicit
  db.commit() on success and db.rollback() on failure. This ensures no partial
  data is ever committed — if the embedding step fails mid-upload, no orphaned
  chunk records are left in the database.

Concurrency
  Uvicorn runs FastAPI in an async event loop. Endpoints that call Ollama or
  perform heavy computation use Python's asyncio to avoid blocking the server
  while waiting for LLM responses. Multiple students can chat simultaneously
  without one request blocking another.

Neo4j Connection Pooling
  The Neo4j Python driver maintains a pool of Bolt connections internally.
  Each request borrows a connection from the pool and returns it when done,
  rather than opening and closing a new TCP connection per query.

Vector Index Performance
  The pgvector ivfflat index on the embedding column allows approximate
  nearest-neighbour search across millions of vector rows in milliseconds.
  Without this index, every query would require a full sequential scan.

Duplicate File Prevention
  When a PDF is uploaded, its SHA-256 hash is computed and compared against
  existing records. If a match is found, the upload is rejected with a clear
  message before any expensive processing begins.


--- Error Handling & Logging ---

All startup events (database connection, Neo4j index creation, Cloudinary
configuration) produce explicit INFO log messages so the operational state
of the system is immediately visible in the terminal.

The /ask SSE endpoint wraps the Ollama call in a try/except. If the model
times out or the connection fails, a structured error event is sent over the
SSE stream to the browser, which displays a user-friendly message instead of
hanging indefinitely.

SlowAPI enforces rate limits per client IP. Exceeding the limit returns a
429 Too Many Requests response with a Retry-After header, preventing a single
user from monopolising Ollama's compute resources.


--- Testing ---

Manual Integration Testing (completed)
  The full user journey has been manually verified end-to-end:
  login as student → submit question → receive streaming answer →
  login as admin → upload PDF → re-query the same content → correct answer returned.
  All API endpoints tested directly via browser DevTools Network panel and cURL.

Planned Automated Testing (Phase 2, documented in enhancements.md)
  pytest unit tests for the Syllabus Parser (test correct node/edge creation
  from known PDF inputs), the embedding pipeline (test vector dimensions and
  non-null output), and the RAG retrieval function (test top-k returns correct
  number of chunks).


=============================================================
SLIDE 4 — Technical Enrichment Activity Choices
=============================================================
Rubric: Individual Mandatory Online Certification — 10 to 20 hours each
Platform must be recognised: Coursera, NPTEL, edX, AWS, Google, NVIDIA, etc.
Course must be relevant to the project domain.

  1. A. Chandra Vamsi       — [Course Title] — [Platform] — [X] Hrs
  2. B. Prem Sai Charan     — [Course Title] — [Platform] — [X] Hrs
  3. M. Pavan Prem Prabhas  — [Course Title] — [Platform] — [X] Hrs
  4. P. Tirupati Reddy      — [Course Title] — [Platform] — [X] Hrs

Suggested courses directly relevant to M.A.C.H.:
  Knowledge Graphs for NLP — Coursera (DeepLearning.AI) — 15 hrs
  Building RAG Agents with LLMs — NVIDIA Deep Learning Institute — 8 hrs
  Vector Databases: from Embeddings to Applications — Coursera — 12 hrs
  LangChain for LLM Application Development — Coursera (DeepLearning.AI) — 10 hrs
  Graph Neural Networks — NPTEL / edX — 20 hrs
  FastAPI Full Course — Udemy / YouTube (not eligible — must be from recognised platform)


=============================================================
SLIDE 5 — References  (IEEE Format)
=============================================================

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,"
    in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[2] Neo4j, Inc., "Neo4j Graph Data Science Library Documentation," neo4j.com, 2024.
    [Online]. Available: https://neo4j.com/docs/graph-data-science/

[3] A. Wang et al., "pgvector: Open-source vector similarity search for Postgres,"
    GitHub, 2024. [Online]. Available: https://github.com/pgvector/pgvector

[4] Ollama, "Run Large Language Models Locally," ollama.com, 2024.
    [Online]. Available: https://ollama.com

[5] S. Ramirez, "FastAPI: Modern, Fast Web Framework for Building APIs with Python,"
    fastapi.tiangolo.com, 2024. [Online]. Available: https://fastapi.tiangolo.com

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding," in Proc. NAACL, 2019.


=============================================================
SLIDE 6 — Thank You
=============================================================

M.A.C.H.
Multimodel Academic Cognitive Hub

Thank You
Questions & Discussion


=============================================================
PRE-REVIEW CHECKLIST
=============================================================

Diagrams to insert:
  [ ] System Architecture Diagram (Browser → FastAPI → PostgreSQL + Neo4j + Ollama + Cloudinary)
  [ ] DB Schema (documents table, document_chunks table, Neo4j node/edge model)
  [ ] RAG Pipeline Flowchart (Query → Classify → Vector + Graph → Merge → LLM → Stream)

Screenshots to capture:
  [ ] Login page with the canvas Knowledge Graph animation running
  [ ] Student chat page showing a real streaming response with source citations
  [ ] Admin dashboard showing the uploaded document list
  [ ] Admin dashboard showing the prerequisite graph visualisation
  [ ] API test evidence (Postman or cURL output for /ask, /upload, /login)

Team details to finalise:
  [ ] Fill in actual MOOC course titles and hours for all 4 team members in Slide 4
  [ ] Add BibTeX entries to ref.bib matching the references above
