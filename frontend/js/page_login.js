/**
 * 🎓 LOGIN PAGE CONTROLLER & KNOWLEDGE GRAPH ANIMATION
 */
import { API_BASE } from './api.js';
import { setAuth } from './auth.js';

/* ── Login ── */
async function doLogin(roleHint) {
  const u = document.getElementById('auth-user').value.trim();
  const p = document.getElementById('auth-pass').value;
  const errEl = document.getElementById('auth-error');
  errEl.textContent = '';

  if (!u || !p) { errEl.textContent = 'Please enter username and password.'; return; }

  const btnS = document.getElementById('btn-student');
  const btnA = document.getElementById('btn-admin');
  btnS.disabled = btnA.disabled = true;

  try {
    const form = new URLSearchParams();
    form.append('username', u);
    form.append('password', p);

    const res = await fetch(`${API_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      errEl.textContent = data.detail || 'Invalid credentials.';
      document.getElementById('auth-user').style.borderColor = 'var(--brand)';
      setTimeout(() => document.getElementById('auth-user').style.borderColor = '', 1400);
      return;
    }

    const data = await res.json();
    const payload = JSON.parse(atob(data.access_token.split('.')[1]));
    const role = payload.role || (u === 'admin' ? 'admin' : 'student');
    
    setAuth(data.access_token, role, u);
    // Redirect based on role
    window.location.href = role === 'admin' ? 'admin.html' : 'student.html';

  } catch (e) {
    errEl.textContent = 'Network error — is the backend running?';
  } finally {
    btnS.disabled = btnA.disabled = false;
  }
}

window.doLogin = doLogin;

document.getElementById('auth-pass').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin('student');
});

// Clear stale token on login page load
localStorage.removeItem('access_token');
localStorage.removeItem('user_role');


/* ══════════════════════════════════════════════════════════════
   KG CANVAS ANIMATION
══════════════════════════════════════════════════════════════ */
const WORD_Q = "question?".split('');
const WORD_A = "answer".split('');

const GRAPH_NODES = [
  { fx: 0.08, fy: 0.12, r: 4.2 }, { fx: 0.46, fy: 0.02, r: 4.6 },
  { fx: 0.88, fy: 0.16, r: 4.2 }, { fx: 0.02, fy: 0.55, r: 4.2 },
  { fx: 0.44, fy: 0.50, r: 6.2 }, { fx: 0.84, fy: 0.54, r: 4.4 },
  { fx: 0.20, fy: 0.92, r: 4.2 }, { fx: 0.64, fy: 0.94, r: 4.4 },
];
const GRAPH_EDGES = [[0,1],[1,2],[0,3],[1,4],[2,5],[3,4],[4,5],[3,6],[4,7],[5,7],[6,7],[0,4]];
const GRAPH_ADJ = (() => {
  const adj = Array.from({ length: GRAPH_NODES.length }, () => []);
  GRAPH_EDGES.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); });
  return adj;
})();

const easeInOut = x => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
const clamp01 = x => Math.min(1, Math.max(0, x));
const seeded = n => { const x = Math.sin(n * 12.9898) * 43758.5453; return x - Math.floor(x); };
const lerp = (a, b, t) => a + (b - a) * t;

function cycleGraphData(ci) {
  const n = GRAPH_NODES.length;
  const scored = [...Array(n).keys()].map(i => ({ i, s: seeded(ci * 13.71 + i * 3.19) }));
  scored.sort((a, b) => b.s - a.s);
  const candidates = scored.slice(0, 5);
  const winner = candidates[0].i;
  const path = [winner];
  let prev = -1, cur = winner;
  for (let h = 0; h < WORD_A.length - 1; h++) {
    const nb = GRAPH_ADJ[cur].filter(x => x !== prev);
    const opts = nb.length ? nb : GRAPH_ADJ[cur];
    const pick = opts[Math.floor(seeded(ci * 5.37 + h * 2.23 + cur * 1.7) * opts.length)];
    path.push(pick); prev = cur; cur = pick;
  }
  return { candidates, winner, path };
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function drawWordBubble(ctx, x, y, text, alpha) {
  if (alpha <= 0.01) return;
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.font = "600 12.5px 'IBM Plex Mono', monospace";
  const tw = ctx.measureText(text).width;
  const bw = tw + 22, bh = 24;
  ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1.2;
  roundRectPath(ctx, x - bw / 2, y - bh / 2, bw, bh, 9); ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.95)';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y + 0.5); ctx.textBaseline = 'alphabetic'; ctx.restore();
}

function drawLetterChip(ctx, x, y, ch, alpha, size) {
  if (alpha <= 0.01) return;
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.font = `500 ${size * 1.05}px 'IBM Plex Mono', monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(ch, x, y + 0.5); ctx.textBaseline = 'alphabetic'; ctx.restore();
}

function drawRAGPipeline(now) {
  const canvas = document.getElementById('kg-canvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);

  const cycle = 15200;
  const ci = Math.floor(now / cycle);
  const phase = (now % cycle) / cycle;
  const { candidates, winner, path } = cycleGraphData(ci);

  const baseY = h * 0.30, midY = h * 0.48;
  const queryX = w * 0.10, rowX = w * 0.24, matrixX = w * 0.35, entryX = w * 0.41, answerX = w * 0.90;
  const gb = { x: w * 0.44, y: h * 0.14, w: w * 0.34, h: h * 0.68 };
  const nodePos = GRAPH_NODES.map(n => ({ x: gb.x + n.fx * gb.w, y: gb.y + n.fy * gb.h, r: n.r }));

  const P = {
    question: [0.00, 0.06], split: [0.06, 0.14], matrix: [0.14, 0.24], condense: [0.24, 0.32],
    match: [0.32, 0.52], traverse: [0.52, 0.82], combine: [0.82, 0.90], hold: [0.90, 1.00],
  };
  const sp = seg => clamp01((phase - seg[0]) / (seg[1] - seg[0]));

  ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(queryX, midY); ctx.lineTo(answerX, midY); ctx.stroke();
  GRAPH_EDGES.forEach(([a, b]) => {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(nodePos[a].x, nodePos[a].y); ctx.lineTo(nodePos[b].x, nodePos[b].y); ctx.stroke();
  });
  nodePos.forEach(n => { ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,255,255,0.10)'; ctx.fill(); });

  const n9 = WORD_Q.length, n6 = WORD_A.length;

  { const qIn = easeInOut(sp(P.question)); const qOut = phase >= P.split[0] ? 1 - easeInOut(sp(P.split)) : 1; const a = Math.min(qIn, qOut); if (a > 0.01) drawWordBubble(ctx, queryX, baseY, "question?", a); }
  if (phase >= P.split[0] && phase < P.matrix[0] + 0.08) { const segP = sp(P.split); for (let i = 0; i < n9; i++) { const li = clamp01((segP - i * 0.05) / (1 - i * 0.05)); if (li <= 0) continue; const ex = easeInOut(li); const tx = rowX - ((n9 - 1) / 2) * 11 + i * 11; const x = lerp(queryX, tx, ex), y = baseY - Math.sin(ex * Math.PI) * 10; const ra = phase >= P.matrix[0] ? 1 - easeInOut(sp(P.matrix)) : 1; drawLetterChip(ctx, x, y, WORD_Q[i], Math.min(li * 1.4, 1) * Math.max(ra, 0), 6.2); } }
  if (phase >= P.matrix[0] && phase < P.condense[0] + 0.08) { const segP = sp(P.matrix); const cell = 12; for (let i = 0; i < n9; i++) { const li = clamp01((segP - i * 0.05) / (1 - i * 0.05)); if (li <= 0) continue; const ex = easeInOut(li); const col = i % 3, row = Math.floor(i / 3); const fx = rowX - ((n9 - 1) / 2) * 11 + i * 11, fy = baseY; const tx = matrixX + (col - 1) * cell, ty = baseY + (row - 1) * cell; const x = lerp(fx, tx, ex), y = lerp(fy, ty, ex); const ca = phase >= P.condense[0] ? 1 - easeInOut(sp(P.condense)) : 1; drawLetterChip(ctx, x, y, WORD_Q[i], ca, 5.6); } if (sp(P.matrix) > 0.5) { ctx.save(); ctx.globalAlpha = clamp01((sp(P.matrix) - 0.5) / 0.5) * (phase >= P.condense[0] ? 1 - easeInOut(sp(P.condense)) : 1); ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 0.8; ctx.strokeRect(matrixX - cell * 1.5 - 2, baseY - cell * 1.5 - 2, cell * 3 + 4, cell * 3 + 4); ctx.restore(); } }
  if (phase >= P.condense[0] && phase < P.match[0] + 0.05) { const ex = easeInOut(sp(P.condense)); ctx.save(); ctx.globalAlpha = 0.5 + ex * 0.45; ctx.beginPath(); ctx.arc(lerp(matrixX, entryX, ex), lerp(baseY, midY, ex), 3, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill(); ctx.restore(); }
  if (phase >= P.match[0] && phase < P.traverse[0] + 0.06) { const segP = sp(P.match); const gP = clamp01(segP / 0.55), nP = clamp01((segP - 0.55) / 0.45); candidates.forEach(c => { const iW = c.i === winner; const t = nodePos[c.i]; const ba = c.s * gP; const a = iW ? Math.max(ba, nP) : ba * (1 - nP); if (a > 0.01) { ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = '#fff'; ctx.lineWidth = iW ? 1 + nP * 1.2 : 1; if (!iW || nP < 0.95) ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(entryX, midY); ctx.lineTo(t.x, t.y); ctx.stroke(); ctx.setLineDash([]); ctx.restore(); } ctx.save(); ctx.globalAlpha = iW ? Math.max(0.4 + ba * 0.5, 0.5 + nP * 0.5) : 0.15 + ba * 0.4 * (1 - nP); ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill(); ctx.restore(); }); ctx.beginPath(); ctx.arc(entryX, midY, 3, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill(); }
  const hops = n6 - 1;
  if (phase >= P.traverse[0] && phase < P.combine[0] + 0.05) { const segP = sp(P.traverse); for (let hi = 0; hi < hops; hi++) { const li = clamp01((segP - hi / hops) / (1 / hops)); if (li <= 0) continue; const a = nodePos[path[hi]], b = nodePos[path[hi + 1]]; const ex = easeInOut(Math.min(1, li)); ctx.save(); ctx.globalAlpha = Math.min(1, li * 1.3); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(a.x + (b.x - a.x) * ex, a.y + (b.y - a.y) * ex); ctx.stroke(); ctx.restore(); } for (let k = 0; k < n6; k++) { const ra = k === 0 ? 0 : k / hops; if (segP < ra) continue; const nd = nodePos[path[k]]; const jr = clamp01((segP - ra) / 0.08); ctx.save(); ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill(); ctx.restore(); drawLetterChip(ctx, nd.x, nd.y - nd.r - 11, WORD_A[k], jr, 6.5); } if (segP > 0.94) { const bp = clamp01((segP - 0.94) / 0.06); const last = nodePos[path[n6 - 1]]; ctx.save(); ctx.globalAlpha = 1 - bp; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(last.x, last.y, last.r + bp * 14, 0, Math.PI * 2); ctx.stroke(); ctx.restore(); } }
  if (phase >= P.combine[0] && phase < P.hold[0] + 0.03) { const ex = easeInOut(sp(P.combine)); for (let k = 0; k < n6; k++) { const nd = nodePos[path[k]]; const x = lerp(nd.x, answerX - ((n6 - 1) / 2) * 10 + k * 10, ex), y = lerp(nd.y - nd.r - 11, baseY, ex); drawLetterChip(ctx, x, y, WORD_A[k], 1 - clamp01((sp(P.combine) - 0.7) / 0.3), 6.5 * (1 - ex * 0.15)); } }
  if (phase >= P.combine[0] + 0.55 * (P.combine[1] - P.combine[0])) { const rs = P.combine[0] + 0.55 * (P.combine[1] - P.combine[0]); const rp = clamp01((phase - rs) / 0.06); const fo = phase >= 0.985 ? clamp01((1 - phase) / 0.015) : 1; drawWordBubble(ctx, answerX, baseY, "answer", rp * fo); }
}

function resizeKG() {
  const canvas = document.getElementById('kg-canvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
}

function kgLoop(now) {
  drawRAGPipeline(now || performance.now());
  requestAnimationFrame(kgLoop);
}

window.addEventListener('resize', resizeKG);
window.addEventListener('load', () => { resizeKG(); kgLoop(); });
