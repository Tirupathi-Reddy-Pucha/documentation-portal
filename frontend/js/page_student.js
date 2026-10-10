/**
 * 🎓 STUDENT DASHBOARD CONTROLLER (RAG INTERFACE)
 * The crown jewel of our FYP frontend. This module orchestrates the chat interface,
 * streaming LLM responses, Markdown parsing, and Voice AI integration.
 * We opted for native Server-Sent Events (SSE) and Streams API instead of bulky libraries like Socket.io.
 */
import { apiFetch } from './api.js';
import { getUsername, requireRole, logout } from './auth.js';
import { timeNow, escapeHtml, simpleMarkdown } from './utils.js';

requireRole('student');

const uname = getUsername() || 'Student';
const initials = uname.slice(0, 2).toUpperCase();

document.getElementById('student-name').textContent = uname.charAt(0).toUpperCase() + uname.slice(1);
document.getElementById('student-avatar').textContent = initials;

let isTyping = false;
let currentSessionId = null;
let chatSessions = {};

function scrollBottom() { 
  const el = document.getElementById('messages'); 
  el.scrollTop = el.scrollHeight; 
}

function saveCurrentSession() {
  if (!currentSessionId) return;
  const msgs = document.getElementById('messages');
  if (chatSessions[currentSessionId]) {
    chatSessions[currentSessionId].html = msgs.innerHTML;
  }
}

const EMPTY_STATE_HTML = `
  <div class="empty-state" id="empty-state">
    <div class="es-glyph">M</div>
    <h3>Ready when you are</h3>
    <p>Ask anything about your syllabus, past year questions, or course outcomes across your entire program. Everything is indexed and searchable.</p>
    <div class="suggestion-grid">
      <button class="sug-chip" onclick="window.sendSuggestion(this)">
        <span class="sug-text">What are the key topics in Unit 2 of Data Structures?</span>
        <span class="sug-icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg></span>
      </button>
      <button class="sug-chip" onclick="window.sendSuggestion(this)">
        <span class="sug-text">Show PYQ questions on database normalization</span>
        <span class="sug-icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></span>
      </button>
      <button class="sug-chip" onclick="window.sendSuggestion(this)">
        <span class="sug-text">Compare course outcomes across two electives</span>
        <span class="sug-icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 17 22 12"/></svg></span>
      </button>
      <button class="sug-chip" onclick="window.sendSuggestion(this)">
        <span class="sug-text">Find every PYQ mentioning a specific topic</span>
        <span class="sug-icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span>
      </button>
    </div>
  </div>`;

export function clearChat() {
  if (isTyping) {
    alert("Please wait for the response to finish before starting a new chat.");
    return;
  }
  saveCurrentSession();
  currentSessionId = crypto.randomUUID();
  chatSessions[currentSessionId] = { id: currentSessionId, title: null, html: EMPTY_STATE_HTML };
  
  document.getElementById('messages').innerHTML = EMPTY_STATE_HTML;
  removeTyping();
  
  document.querySelectorAll('.hist-item').forEach(h => h.classList.remove('active'));
}
window.clearChat = clearChat;

export function loadChat(id) {
  if (isTyping) {
    alert("Please wait for the response to finish before switching chats.");
    return;
  }
  if (currentSessionId === id) return;
  
  saveCurrentSession();
  currentSessionId = id;
  const session = chatSessions[id];
  document.getElementById('messages').innerHTML = session.html;
  
  document.querySelectorAll('.hist-item').forEach(h => h.classList.remove('active'));
  const activeItem = document.getElementById('hist-' + id);
  if (activeItem) activeItem.classList.add('active');
  
  scrollBottom();
}
window.loadChat = loadChat;

clearChat();

function addUserMessage(text) {
  const es = document.getElementById('empty-state');
  if (es) es.remove();
  const row = document.createElement('div');
  row.className = 'msg-row user-row';
  row.innerHTML = `
    <div class="msg-av user">${initials}</div>
    <div class="msg-content">
      <div class="bubble">${escapeHtml(text)}</div>
      <div class="bubble-footer">
        <div></div>
        <div class="bubble-time">${timeNow()}</div>
      </div>
    </div>`;
  document.getElementById('messages').appendChild(row);
  scrollBottom();
}

function showTyping() {
  if (isTyping) return;
  isTyping = true;
  const row = document.createElement('div');
  row.className = 'typing-row'; row.id = 'typing-row';
  row.innerHTML = `<div class="msg-av ai">M</div><div class="typing-bubble"><div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div></div>`;
  document.getElementById('messages').appendChild(row);
  scrollBottom();
}

function removeTyping() {
  const tr = document.getElementById('typing-row');
  if (tr) tr.remove();
  isTyping = false;
}

function createAIBubble(msgId) {
  const id = 'msg-' + Date.now();
  const safeId = msgId || '';
  const row = document.createElement('div');
  row.className = 'msg-row';
  row.innerHTML = `
    <div class="msg-av ai">M</div>
    <div class="msg-content">
      <div class="bubble ai-bubble" id="${id}"></div>
      <div class="bubble-footer">
        <div class="bubble-time">M.A.C.H. · ${timeNow()}</div>
        <div class="feedback-btns">
          <button class="fb-btn" onclick="window.vote(this,'up','${safeId}')" title="Helpful">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>
          </button>
          <button class="fb-btn" onclick="window.vote(this,'down','${safeId}')" title="Not helpful">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
        </div>
      </div>
    </div>`;
  document.getElementById('messages').appendChild(row);
  scrollBottom();
  return document.getElementById(id);
}

/**
 * Renders a structured "Problem List" response as visual cards.
 * Called when the backend returns X-Response-Type: problem_list (PROBLEM_LIST intent).
 * Each card shows: question number, marks, BTL level, and the question text.
 *
 * @param {object} data - The parsed JSON from the /chat endpoint
 * @param {string} data.topic - The extracted topic name
 * @param {Array}  data.problems - Array of problem objects from Neo4j
 * @returns {string} Safe HTML string
 */
function renderProblemCards(data) {
  const { topic = 'Topic', problems = [] } = data;

  if (!problems.length) {
    return `
      <div class="problem-list-empty">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        <p>No problems found for <strong>${escapeHtml(topic)}</strong>. Try uploading more PYQ papers for this topic.</p>
      </div>`;
  }

  const btlLabel = (btl) => {
    const map = { 1:'Remember', 2:'Understand', 3:'Apply', 4:'Analyse', 5:'Evaluate', 6:'Create' };
    const n = parseInt(btl);
    return map[n] ? `BTL ${n} · ${map[n]}` : `BTL ${btl}`;
  };

  const cards = problems.map((p, idx) => {
    const qText   = escapeHtml(p.text || p.question_text || 'Question text unavailable');
    const marks   = p.marks   ? `<span class="pc-mark">${p.marks} marks</span>` : '';
    const btl     = p.btl    ? `<span class="pc-btl">${btlLabel(p.btl)}</span>` : '';
    const course  = p.course_code ? `<span class="pc-course">${escapeHtml(p.course_code)}</span>` : '';
    const imgHtml = p.image_url && p.image_url !== 'None'
      ? `<img src="${escapeHtml(p.image_url)}" alt="Diagram for Q${idx+1}" class="pc-img">`
      : '';
      
    // Graph RAG indicators
    let aiBadge = '';
    if (p.mapping_confidence) {
      const conf = Math.round(p.mapping_confidence * 100);
      aiBadge = `<span class="pc-ai-badge" title="AI Confidence: ${conf}% mapped to topic: ${p.topic_name}">
        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:4px"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        AI Mapped
      </span>`;
    }

    return `
      <div class="problem-card">
        <div class="pc-header">
          <span class="pc-num">Q${idx + 1}</span>
          <div class="pc-tags">${aiBadge}${course}${marks}${btl}</div>
        </div>
        <div class="pc-body">${qText}</div>
        ${imgHtml}
      </div>`;
  }).join('');

  return `
    <div class="problem-list-wrap">
      <div class="problem-list-header">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>Problems on <strong>${escapeHtml(topic)}</strong></span>
        <span class="pl-count">${problems.length} found</span>
      </div>
      <div class="problem-cards">${cards}</div>
    </div>`;
}

export async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || isTyping) return;

  input.value = ''; input.style.height = '';
  addUserMessage(text);
  addToHistory(text);
  document.getElementById('send-btn').disabled = true;
  showTyping();

  try {
    const res = await apiFetch('/chat', {
      method: 'POST',
      body: JSON.stringify({ message: text, session_id: currentSessionId }),
    });

    // ── Check if backend returned a structured Problem List (not a streaming text) ──
    const responseType = res.headers.get('X-Response-Type');
    if (responseType === 'problem_list') {
      const data = await res.json();
      removeTyping();
      const bubble = createAIBubble(null);
      bubble.innerHTML = renderProblemCards(data);
      scrollBottom();
      return;  // skip streaming logic
    }

    // ── Normal streaming text response ──
    const msgId = res.headers.get('X-Message-ID') || null;
    removeTyping();
    const bubble = createAIBubble(msgId);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      fullText += decoder.decode(value, { stream: true });
      let rawHtml = window.marked ? window.marked.parse(fullText) : simpleMarkdown(fullText);
      let safeHtml = window.DOMPurify ? window.DOMPurify.sanitize(rawHtml) : rawHtml;
      bubble.innerHTML = safeHtml;
      scrollBottom();
    }
    
    // Speak the AI's response aloud
    if ('speechSynthesis' in window) {
      // Strip markdown characters so it reads naturally
      const plainText = fullText.replace(/[*#_`]/g, '').replace(/\n/g, '. ');
      const utterance = new SpeechSynthesisUtterance(plainText);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    if (msgId) {
      bubble.parentElement.querySelectorAll('.fb-btn').forEach(btn => {
        const dir = btn.getAttribute('onclick').includes("'up'") ? 'up' : 'down';
        btn.setAttribute('onclick', `vote(this,'${dir}','${msgId}')`);
      });
    }

  } catch (e) {
    removeTyping();
    if (e.message !== "Unauthorized") {
      createAIBubble(null).innerHTML = '<em style="color:var(--brand)">Network error or backend failed.</em>';
    }
  } finally {
    document.getElementById('send-btn').disabled = false;
    isTyping = false;
  }
}
window.sendMessage = sendMessage;

export function sendSuggestion(btn) {
  document.getElementById('chat-input').value = btn.querySelector('.sug-text').textContent;
  sendMessage();
}
window.sendSuggestion = sendSuggestion;

export async function vote(btn, dir, msgId) {
  btn.parentElement.querySelectorAll('.fb-btn').forEach(b => b.classList.remove('voted-up', 'voted-down'));
  btn.classList.add(dir === 'up' ? 'voted-up' : 'voted-down');

  if (!msgId) return;
  try {
    await apiFetch(`/feedback/${msgId}`, {
      method: 'POST',
      body: JSON.stringify({ score: dir === 'up' ? 1 : -1 }),
    });
  } catch (e) { /* silent */ }
}
window.vote = vote;

function addToHistory(text) {
  if (!chatSessions[currentSessionId].title) {
    chatSessions[currentSessionId].title = text;
    updateHistoryUI(currentSessionId, text);
  }
}

function updateHistoryUI(id, text) {
  const hist = document.getElementById('sidebar-history');
  let item = document.getElementById('hist-' + id);
  if (!item) {
    item = document.createElement('div');
    item.id = 'hist-' + id;
    item.className = 'hist-item active';
    item.setAttribute('onclick', `window.loadChat('${id}')`);
    hist.prepend(item);
  }
  
  hist.querySelectorAll('.hist-item').forEach(h => h.classList.remove('active'));
  item.classList.add('active');
  
  item.innerHTML = `<span class="hist-icon"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg></span><span class="hist-text">${escapeHtml(text.slice(0, 48))}${text.length > 48 ? '…' : ''}</span>`;
}

document.getElementById('chat-input').addEventListener('input', function () {
  this.style.height = '';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

document.getElementById('chat-input').addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

export function handleAttach(input) {
  if (input.files[0]) {
    document.getElementById('chat-input').value = `[Attached: ${input.files[0].name}] `;
    document.getElementById('chat-input').focus();
  }
}
window.handleAttach = handleAttach;
window.logout = logout;

// Voice AI Integration (Speech-to-Text)
let recognition = null;
let isListening = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isListening = true;
    document.getElementById('mic-btn').classList.add('listening');
    document.getElementById('chat-input').placeholder = "Listening...";
    // Cancel any ongoing speech synthesis if user interrupts
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('chat-input').value = transcript;
    window.sendMessage();
  };

  recognition.onerror = (e) => {
    console.error("Speech recognition error:", e);
    stopListening();
  };

  recognition.onend = () => {
    stopListening();
  };
}

function stopListening() {
  isListening = false;
  document.getElementById('mic-btn').classList.remove('listening');
  document.getElementById('chat-input').placeholder = "Ask about your syllabus, PYQs, or any topic…";
}

export function toggleSpeech() {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser. Try Google Chrome.");
    return;
  }
  if (isListening) {
    recognition.stop();
  } else {
    recognition.start();
  }
}
window.toggleSpeech = toggleSpeech;

