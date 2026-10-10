# M.A.C.H. 
**Multimodel Academic Cognitive Hub - A Fine-tuned Graph-RAG Framework for Campus Knowledge Systems.**

A Graph-RAG system that turns university syllabi, regulations, timetables, and past year question papers into an interactive Knowledge Graph, allowing students to ask natural language questions grounded in official course materials.

---

## 📁 Repository Structure

```
MACH/
├── README.md               # Main project documentation & setup guide
├── Dockerfile              # Docker container definition for backend
├── docker-compose.yml      # PostgreSQL (pgvector) + Neo4j container setup
├── requirements.txt        # Python dependencies
│
├── backend/                # FastAPI Application
│   ├── main.py             # Server entrypoint, CORS, background tasks
│   ├── api/routers/        # Modular route endpoints
│   ├── core/
│   │   └── config.py       # Pydantic Settings configuration
│   ├── db/
│   │   ├── database.py     # PostgreSQL & Neo4j database connections
│   │   └── query_neo4j.py  # Knowledge graph Cypher query handlers
│   ├── services/
│   │   ├── utils.py        # PDF slicing, embeddings, vision AI
│   │   └── syllabus_parser.py # PDF text extraction logic
│   └── cache/              # Automatically excluded cache files (__pycache__, .ruff_cache)
│
├── frontend/               # Web Application Interface (PWA)
│   ├── index.html          # Main login page
│   ├── student.html        # Student chat & problem card portal
│   ├── admin.html          # Admin dashboard & upload manager
│   └── src/                # Modular CSS & JavaScript modules
│
└── docs/                   # Project Documentation
    └── enhancements.md     # Current implementation plan and ideas
```

---

## ⚡ Quickstart Setup

### Requirements
- Docker Desktop (for PostgreSQL & Neo4j)
- Python 3.11+
- Ollama (Local or Colab-tunneled via Ngrok)

### Step 1 — Configure Environment
Copy `.env.example` to `.env` in the project root:
```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:12b-it-qat
OLLAMA_EMBED_MODEL=nomic-embed-text

POSTGRES_USER=cluser
POSTGRES_PASSWORD=clpassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_DB=curriculumlens

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=clpassword
```

### Step 2 — Start Databases (Docker)
```bash
docker compose up -d
```

### Step 3 — Virtual Environment & Dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Run Backend Server
```bash
cd backend
python main.py
```
The server will start at: `http://localhost:8000`

### Step 5 — Open Application
Access the app in your browser at:
**`http://localhost:8000/public/index.html`**

- **Student Portal**: Sign in with `student` / `student123`
- **Admin Panel**: Sign in with `admin` / `admin123`

---

## ☁️ Google Colab GPU Setup (Optional)

If running Ollama locally is slow, run it on a free Google Colab T4 GPU:

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook with **T4 GPU** accelerator (**Runtime > Change runtime type**).
2. Execute the following script in a Colab code cell (replace `YOUR_NGROK_TOKEN` from [ngrok.com](https://dashboard.ngrok.com)):

```python
!apt-get update -qq && apt-get install -y -qq zstd pciutils > /dev/null 2>&1
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
!pip install pyngrok -q > /dev/null 2>&1

import os, threading, time
from pyngrok import ngrok

threading.Thread(target=lambda: os.system("OLLAMA_HOST=0.0.0.0 ollama serve"), daemon=True).start()
time.sleep(3)

!ollama pull nomic-embed-text > /dev/null 2>&1
!ollama pull gemma4:12b-it-qat > /dev/null 2>&1

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(11434).public_url
print(f"\n✅ OLLAMA URL: {public_url}\n")
```

3. Copy the generated `https://xxxx.ngrok-free.app` URL into your local `.env` file as `OLLAMA_BASE_URL`.

---

## 🛠️ Troubleshooting

- **"Connection refused" / DB Errors**: Ensure Docker Desktop is running and run `docker compose up -d`.
- **"Failed to fetch" on Login/Chat**: Ensure the backend server (`python main.py`) is running on port 8000.
- **Ollama Timeout**: If using Colab, ensure your Ngrok tunnel URL is active and updated in `.env`.
- **System Reset**: In the Admin Panel, scroll to System Settings and click **Reset Everything** to clear database states.
