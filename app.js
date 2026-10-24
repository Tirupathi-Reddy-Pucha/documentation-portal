/**
 * M.A.C.H. - Multimodal Academic Cognitive Hub
 * Official Academic Research & Technical Documentation Portal Engine
 * Publication-Grade Light Theme Implementation
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initArchitectureInspector();
  initDemoShowcasePlayer();
  initDocsPortal();
  initCodeCopyButtons();
});

/* ==========================================================================
   1. Smooth Navigation & Active States
   ========================================================================== */

function initNavScroll() {
  const navLinks = document.querySelectorAll('.nav-menu a');
  const sections = document.querySelectorAll('section[id]');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.scrollY + 100;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = sec.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   2. Interactive Architecture Visualizer
   ========================================================================== */

const ARCH_COMPONENTS = {
  client: {
    title: 'Student UI (ES6 Native PWA)',
    file: 'frontend/src/js/pages/student.js & admin.js',
    role: 'Zero-build native ES module client providing real-time streaming markdown, Cytoscape/Vis.js graph visualization, and PDF ingestion.',
    latency: '< 10 ms local client render',
    code: `// frontend/src/js/pages/student.js
const response = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userQuery, session_id: sid })
});
renderStream(response.body);`
  },
  gateway: {
    title: 'FastAPI High-Throughput ASGI Gateway',
    file: 'backend/app.py:9-1845',
    role: 'Asynchronous router orchestrating JWT auth, query rewriting, intent classification dispatch, and streaming server-sent events (SSE).',
    latency: 'Sub-5ms async dispatch',
    code: `@app.post("/chat")
async def chat_endpoint(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    rewritten = await rewrite_query(req.message, req.history)
    intent = await classify_intent_llm(rewritten)
    return await dispatch_intent(intent, rewritten, req)`
  },
  engine1: {
    title: 'Engine 1: Vision Subsystem (gemma4:12b-it-qat)',
    file: 'backend/vision_query.py & utils.py',
    role: 'Dedicated Google Gemma-4 (12B QAT) engine invoked exclusively for OCR, mathematical derivations, and circuit schematic analysis. Eliminates multimodal interference on text reasoning.',
    latency: '~1.8s - 2.5s (High-resolution multimodal OCR pass)',
    code: `# backend/vision_query.py
async def describe_page_image(image_bytes: bytes) -> str:
    response = await ollama_client.generate(
        model=settings.OLLAMA_VISION_MODEL, # "gemma4:12b-it-qat"
        prompt="Extract all circuit diagrams, schematics, and equations...",
        images=[base64.b64encode(image_bytes).decode()]
    )
    return response['response']`
  },
  engine2: {
    title: 'Engine 2: Core SLM (campus-llm 3B)',
    file: 'campus-llm_gguf/Qwen2.5-3B-Instruct.Q4_K_M.gguf',
    role: 'Fine-tuned Qwen2.5-3B model via 4-bit QLoRA. Executes all real-time text reasoning: sub-150ms intent classification, multi-turn query rewriting, knowledge graph triplet extraction, and monotonic Graph-RAG generation.',
    latency: '148 ms (Intent) | 182 ms (Rewriting) | 99.8% Schema Compliance',
    code: `FROM Qwen2.5-3B-Instruct.Q4_K_M.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
PARAMETER num_predict 2048`
  },
  neo4j: {
    title: 'Neo4j 5.24 Knowledge Graph',
    file: 'backend/query_neo4j.py & backend/migrations/neo4j_schema.py',
    role: 'Index-free adjacency graph store maintaining Courses, Units, Topics, Prerequisites, Bloom Taxonomy Levels, and PYQ Question links.',
    latency: '8 - 25 ms Cypher traversals',
    code: `// Fetch hierarchical curriculum with monotonic unit order
MATCH (c:Course {code: $course_code})-[:CONTAINS]->(u:Unit)
OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
OPTIONAL MATCH (t)<-[:TESTS_TOPIC]-(q:Question)
RETURN u.number AS unit_num, u.title AS unit_title, 
       collect({topic: t.name, bloom: t.bloom_level}) AS topics
ORDER BY unit_num ASC;`
  },
  postgres: {
    title: 'PostgreSQL 16 + pgvector + tsvector (RRF)',
    file: 'backend/database.py & utils.py (hybrid_search_rrf)',
    role: 'Hybrid retrieval engine fusing 768-dim dense embeddings (nomic-embed-text) with lexical English full-text search (tsvector) using Reciprocal Rank Fusion (k=60).',
    latency: '15 - 35 ms HNSW cosine search',
    code: `-- Hybrid RRF Query
WITH dense AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $vector) AS rank_dense
  FROM document_chunks ORDER BY embedding <=> $vector LIMIT 30
),
lexical AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(search_vector, query) DESC) AS rank_lex
  FROM document_chunks, to_tsquery('english', $text) query
  WHERE search_vector @@ query LIMIT 30
)
SELECT COALESCE(d.id, l.id) AS chunk_id,
       COALESCE(1.0 / (60 + d.rank_dense), 0.0) + COALESCE(1.0 / (60 + l.rank_lex), 0.0) AS rrf_score
FROM dense d FULL OUTER JOIN lexical l ON d.id = l.id
ORDER BY rrf_score DESC LIMIT 10;`
  },
  slicer: {
    title: 'Two-Pass PyMuPDF Geometric Slicer',
    file: 'backend/utils.py:219-380 & pyq_processor.py',
    role: 'High-precision coordinate extraction for examination question papers. Computes vertical Y-axis bounding boxes, detects question continuations across page breaks, and crops schematic diagrams at 2x resolution.',
    latency: '45 ms per PDF page',
    code: `# backend/utils.py
rect = fitz.Rect(0, start_y * scale, page.rect.width * scale, end_y * scale)
pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2.0, 2.0))
img = Image.open(io.BytesIO(pix.tobytes("png")))
cdn_url = await upload_to_cloudinary(img)`
  }
};

function initArchitectureInspector() {
  const nodes = document.querySelectorAll('.arch-node-clickable');
  const titleEl = document.getElementById('inspectorTitle');
  const fileEl = document.getElementById('inspectorFile');
  const roleEl = document.getElementById('inspectorRole');
  const latencyEl = document.getElementById('inspectorLatency');
  const codeEl = document.getElementById('inspectorCode');

  if (!nodes.length || !titleEl) return;

  function selectComponent(key) {
    const data = ARCH_COMPONENTS[key];
    if (!data) return;

    nodes.forEach(node => {
      const rect = node.querySelector('rect');
      if (node.getAttribute('data-component') === key) {
        if (rect) rect.classList.add('selected');
      } else {
        if (rect) rect.classList.remove('selected');
      }
    });

    titleEl.textContent = data.title;
    fileEl.textContent = data.file;
    roleEl.textContent = data.role;
    latencyEl.textContent = data.latency;
    codeEl.textContent = data.code;
  }

  nodes.forEach(node => {
    node.addEventListener('click', () => {
      const compKey = node.getAttribute('data-component');
      selectComponent(compKey);
    });
  });

  selectComponent('engine2');
}

/* ==========================================================================
   3. Formal Demonstration Showcase Player
   ========================================================================== */

const DEMO_SCENARIOS = [
  {
    name: 'Demo 1: Conversational Graph-RAG Student Chat',
    duration: '02:15',
    steps: [
      {
        time: '00:10',
        log: '[148ms] campus-llm classified student query: "what do I study in year 1" -> Intent: SEMESTER_COURSES (Confidence: 0.994)',
        query: 'what do I study in year 1',
        tag: 'Intent: SEMESTER_COURSES &bull; Latency: 148 ms',
        response: `Curriculum Structure for First Year (B.Tech Computer Science & Engineering):
• Semester 1: 23MAT101 Single Variable Calculus, 23CSE101 Problem Solving & Algorithmic Thinking, 23ENG101 Technical Communication.
• Semester 2: 23MAT102 Multivariable Calculus, 23CSE102 Data Structures & Algorithms, 23PHY101 Engineering Physics.`
      },
      {
        time: '00:45',
        log: '[182ms] Query Rewriter resolved pronoun anaphora: "what are the textbooks for it?" -> "what are the prescribed textbooks for Data Structures & Algorithms"',
        query: 'what are the textbooks for it?',
        tag: 'Anaphora Resolution &bull; Context Grounded &bull; Latency: 182 ms',
        response: `Official Prescribed Textbooks for 23CSE102 (Data Structures & Algorithms):
1. Introduction to Algorithms, T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein (CLRS), 3rd Edition, MIT Press.
2. Data Structures and Algorithm Analysis in C++, Mark Allen Weiss, 4th Edition, Pearson Education.
(Derived strictly from Syllabus Unit 1-5 official bibliography)`
      },
      {
        time: '01:20',
        log: '[142ms] Intent: PROBLEM_LIST -> Neo4j traversal: MATCH (t:Topic {name: "Binary Search Trees"})<-[:TESTS_TOPIC]-(q:Question)',
        query: 'give me exam questions with diagrams on binary search trees',
        tag: 'Graph Traversal: TESTS_TOPIC &bull; Bloom Level: L4 Analyze',
        response: `Verified Examination Question [Mid-Term 2024 - 5 Marks - Bloom: L4 Analyze]:
"Construct an AVL tree by sequentially inserting: [45, 23, 76, 12, 34, 89, 90]. Demonstrate the double rotation steps required to maintain balance factor invariants."
Diagram Reference: /pyq/23CSE102/2024_midterm_q3b.png (Schematic coordinate box: 972x340 px)`
      }
    ]
  },
  {
    name: 'Demo 2: Physical Layout PYQ Slicing & Schematics',
    duration: '01:45',
    steps: [
      {
        time: '00:15',
        log: '[45ms] PyMuPDF rasterizing exam paper: 23CSE302_MidTerm_2024.pdf at 2x resolution (300 DPI)',
        query: 'Ingestion of 23CSE302_EndSem_2024.pdf (Page 2 of 4)',
        tag: 'Coordinate Slicer: Header Regex ^(\\d+)\\s*\\.',
        response: `Detected Question Headers:
• Q4(a): "Analyze the following circuit diagram and determine the state transition table..." [Y-Coords: 142.5 -> 388.2]
• Sub-diagram detected within rect: (x0=54, y0=190, x1=540, y1=360). Dynamic enclosure expanded by 15px.
Cloudinary Upload: /schematics/23CSE302_q4a.png`
      }
    ]
  },
  {
    name: 'Demo 3: Knowledge Graph Explorer & Prerequisite Paths',
    duration: '02:00',
    steps: [
      {
        time: '00:18',
        log: '[18ms] Cypher: MATCH (c:Course {code: "23CSE331"})-[:CONTAINS]->(u:Unit)-[:HAS_TOPIC]->(t:Topic) RETURN c, u, t',
        query: 'Traversing Knowledge Graph for 23CSE331 (Cryptography)',
        tag: 'Neo4j Bolt: bolt://localhost:7687 &bull; 1 Course, 5 Units, 28 Topics',
        response: `Hierarchical Entity Adjacency Tree:
• Course: 23CSE331 Cryptography and Network Security
  ├── Unit 1: Classical Ciphers & Modular Arithmetic (Bloom: L2, L3)
  ├── Unit 2: Symmetric Key Cryptography - AES & DES (Bloom: L3, L4)
  ├── Unit 3: Asymmetric Cryptography - RSA, ECC & Diffie-Hellman (Bloom: L4, L5)
  ├── Unit 4: Hash Functions & Digital Signatures (Bloom: L4)
  └── Unit 5: Network Security Protocols - TLS 1.3, IPsec (Bloom: L4, L6)`
      }
    ]
  },
  {
    name: 'Demo 4: Admin Ingestion & Dual-Engine Routing',
    duration: '02:30',
    steps: [
      {
        time: '00:20',
        log: '[148ms] Document Dispatch: Pages 1-3 routed to campus-llm 3B; Page 4 (Scanned Schematic) routed to Vision Engine (gemma4 12B)',
        query: 'Ingesting Amrita B.Tech CSE Curriculum Regulations PDF',
        tag: 'Decoupled Dual-Engine Routing &bull; Zero Multimodal Degradation',
        response: `Ingestion Report:
1. campus-llm (3B) parsed 42 syllabus topics and Bloom taxonomy ratings with 100% strict JSON compliance in 1.4 seconds.
2. Vision Engine (gemma4:12b) processed circuit schematics from Appendix B without memory contention.
3. PostgreSQL pgvector indexed 768-dim embeddings via nomic-embed-text.`
      }
    ]
  }
];

function initDemoShowcasePlayer() {
  const tabs = document.querySelectorAll('.demo-tab-btn');
  const playBtn = document.getElementById('demoPlayBtn');
  const scrubberBar = document.getElementById('demoProgressBar');
  const timestampEl = document.getElementById('demoTimestamp');
  const telemetryEl = document.getElementById('demoTelemetryLog');
  const messageListEl = document.getElementById('simMessageList');
  const speedSelector = document.getElementById('demoSpeedSelector');

  if (!tabs.length || !messageListEl) return;

  let currentScenarioIndex = 0;
  let currentStepIndex = 0;
  let isPlaying = false;
  let playbackInterval = null;
  let playbackSpeed = 1.0;

  function renderScenario(scenarioIndex, stepIndex) {
    const scenario = DEMO_SCENARIOS[scenarioIndex];
    if (!scenario) return;

    const step = scenario.steps[stepIndex] || scenario.steps[0];
    
    if (telemetryEl) {
      telemetryEl.textContent = step.log;
    }

    if (timestampEl) {
      timestampEl.textContent = `${step.time} / ${scenario.duration}`;
    }

    if (scrubberBar) {
      const pct = Math.min(100, Math.round(((stepIndex + 1) / scenario.steps.length) * 100));
      scrubberBar.style.width = `${pct}%`;
    }

    messageListEl.innerHTML = `
      <div class="trace-query-box">
        <strong>Input Query / Operation:</strong>
        ${escapeHtml(step.query)}
      </div>
      <div class="trace-result-box">
        <span class="trace-meta-tag">${step.tag}</span>
        <div style="white-space: pre-line;">${escapeHtml(step.response)}</div>
      </div>
    `;
  }

  function startPlayback() {
    isPlaying = true;
    if (playBtn) playBtn.textContent = 'Pause Trace';
    
    clearInterval(playbackInterval);
    const intervalMs = Math.round(3000 / playbackSpeed);

    playbackInterval = setInterval(() => {
      const scenario = DEMO_SCENARIOS[currentScenarioIndex];
      currentStepIndex++;
      if (currentStepIndex >= scenario.steps.length) {
        currentStepIndex = 0;
      }
      renderScenario(currentScenarioIndex, currentStepIndex);
    }, intervalMs);
  }

  function pausePlayback() {
    isPlaying = false;
    if (playBtn) playBtn.textContent = 'Play Trace';
    clearInterval(playbackInterval);
  }

  if (playBtn) {
    playBtn.addEventListener('click', () => {
      if (isPlaying) {
        pausePlayback();
      } else {
        startPlayback();
      }
    });
  }

  if (speedSelector) {
    speedSelector.addEventListener('change', (e) => {
      playbackSpeed = parseFloat(e.target.value);
      if (isPlaying) {
        startPlayback();
      }
    });
  }

  tabs.forEach((tab, idx) => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentScenarioIndex = idx;
      currentStepIndex = 0;
      renderScenario(currentScenarioIndex, currentStepIndex);
      if (isPlaying) {
        startPlayback();
      }
    });
  });

  renderScenario(0, 0);
}

/* ==========================================================================
   4. Documentation Portal Chapters & Search
   ========================================================================== */

function initDocsPortal() {
  const searchInput = document.getElementById('docsSearchInput');
  const navItems = document.querySelectorAll('.docs-sidebar-nav li');
  const chapters = document.querySelectorAll('.doc-chapter-article');

  if (!navItems.length) return;

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = item.getAttribute('data-chapter');

      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      chapters.forEach(chap => {
        if (chap.getAttribute('id') === targetId) {
          chap.style.display = 'block';
        } else {
          chap.style.display = 'none';
        }
      });
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase().trim();
      navItems.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(query) || query === '') {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      });
    });
  }
}

/* ==========================================================================
   5. Copy-to-Clipboard Utility
   ========================================================================== */

function initCodeCopyButtons() {
  const copyBtns = document.querySelectorAll('.btn-copy-code');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const codeBlock = btn.closest('.doc-code-academic') || btn.closest('.inspector-code-box');
      if (!codeBlock) return;

      const codeText = codeBlock.querySelector('.code-pre-block') || codeBlock.querySelector('pre') || codeBlock;
      if (!codeText) return;

      navigator.clipboard.writeText(codeText.innerText.trim()).then(() => {
        const originalText = btn.textContent;
        btn.textContent = 'Copied';
        btn.style.borderColor = '#1b5e20';
        setTimeout(() => {
          btn.textContent = originalText;
          btn.style.borderColor = '';
        }, 1800);
      }).catch(err => {
        console.error('Copy failed: ', err);
      });
    });
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
