# M.A.C.H. - Future Enhancements Plan

This document outlines the roadmap for upgrading M.A.C.H. (Multimodel Academic Cognitive Hub) into a production-ready, enterprise-grade application. These enhancements cover architecture, AI improvements, and frontend modernization.

---

## Phase 1: Core System Modernization

### 1. Full API Modularization
**Goal:** Split the massive 1,300-line `main.py` monolith into logical, maintainable endpoints.
- **Implementation:** Utilize FastAPI's `APIRouter` to break routes into `api/routers/auth.py`, `chat.py`, `upload.py`, and `system.py`.

### 2. Database Migrations (Alembic)
**Goal:** Safely track and apply changes to the PostgreSQL schema without losing user data (replacing `Base.metadata.create_all`).
- **Implementation:** Integrate `alembic` to generate and apply migration scripts for any future table or column additions.

### 3. Automated Testing (Pytest)
**Goal:** Ensure system stability and prevent regressions as the project scales.
- **Implementation:** Write unit tests using `pytest` for critical paths like PDF parsing, Vision AI extraction, and RAG retrieval.

---

## Phase 2: AI & Knowledge Graph Enhancements

### 1. Conversation Memory
**Goal:** Allow the AI to remember context across multiple messages in a session (e.g., answering "Can you explain that last point?").
- **Implementation:** Store chat history in PostgreSQL or Redis and inject the last N messages into the Ollama prompt context.

### 2. LLM Orchestration (LlamaIndex)
**Goal:** Standardize document chunking, prompt formatting, and interaction with Ollama.
- **Implementation:** Replace manual prompt formatting with LlamaIndex's robust `Ollama` LLM class and `PromptTemplate` for better Retrieval-Augmented Generation (RAG).

---

## Phase 3: Robust Architecture

### 1. Background Processing (Celery + Redis)
**Goal:** Prevent heavy PDF vision extraction from blocking the FastAPI server and handle crash recovery.
- **Implementation:** Spin up a Redis container and replace FastAPI's `BackgroundTasks` with Celery workers (`.delay()`).

### 2. Authentication (Supabase Auth)
**Goal:** Replace hardcoded JWTs with enterprise-grade, secure authentication.
- **Implementation:** Implement Supabase Auth UI (supporting email/password, Google, etc.) and verify JWTs on the FastAPI backend using public keys.

---

## Phase 4: Frontend Migration (Next.js)

### 1. Next.js & React Refactor
**Goal:** Move from Vanilla JS/HTML to a modern state-managed framework for robust chat streaming and UI stability, **without changing a single pixel of the current design.**
- **Implementation:** 
  - Initialize a Next.js project.
  - Port existing CSS and UI elements into React components (`ChatLayout.tsx`).
  - Port the raw Canvas Knowledge Graph animation into a React `useEffect` hook.
  - Establish a central API client to communicate with the FastAPI backend.

> [!TIP]
> **Design Guarantee:** The hard work put into the UI will NOT be lost. The custom layouts, animations, and typography will look exactly the same—only the underlying code managing the state will change.
