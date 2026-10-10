import { apiFetch } from './api.js';
import { getUsername, requireRole, logout } from './auth.js';

requireRole('admin');

const uname = getUsername() || 'Admin';
document.getElementById('admin-name').textContent = uname.charAt(0).toUpperCase() + uname.slice(1);

export function adminNav(el, section) {
  document.querySelectorAll('.a-nav-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');

  const sections = ['overview', 'upload', 'eval', 'settings'];
  sections.forEach(s => {
    const sectionEl = document.getElementById(`section-${s}`);
    if (sectionEl) sectionEl.style.display = 'none';
  });

  const activeSection = document.getElementById(`section-${section}`);
  if (activeSection) activeSection.style.display = 'block';
}
window.adminNav = adminNav;

let uploadTimer = null;

export function dzOver(e) { e.preventDefault(); document.getElementById('dropzone').classList.add('over'); }
export function dzLeave() { document.getElementById('dropzone').classList.remove('over'); }
export function dzDrop(e) {
  e.preventDefault(); dzLeave();
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    document.getElementById('file-input').files = files; // Works in modern browsers or we just keep reference
    document.getElementById('file-input')._droppedFiles = files;
    updateFileText(files);
  }
}
window.dzOver = dzOver;
window.dzLeave = dzLeave;
window.dzDrop = dzDrop;

function updateFileText(files) {
  const textEl = document.getElementById('dz-filename');
  if (files.length === 1) textEl.textContent = files[0].name;
  else if (files.length > 1) textEl.textContent = `${files.length} files selected`;
  else textEl.textContent = '';
}

export function fileSelected(input) {
  updateFileText(input.files);
}
window.fileSelected = fileSelected;

export function docTypeChanged() {
  const docType = document.getElementById('doc-type').value;
  document.getElementById('course-code-field').style.display = docType === 'pyq' ? 'block' : 'none';
  document.getElementById('dept-field').style.display = docType === 'syllabus' ? 'block' : 'none';
  document.getElementById('year-field').style.display = docType === 'syllabus' ? 'block' : 'none';
}
window.docTypeChanged = docTypeChanged;

function setProg(pct, label) {
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-pct').textContent = pct + '%';
  document.getElementById('prog-label').textContent = label;
}

export async function doUpload() {
  const docType = document.getElementById('doc-type').value;
  const fileInput = document.getElementById('file-input');
  const files = fileInput._droppedFiles || fileInput.files;

  let code = '';
  let dept = '';
  let year = '';

  if (docType === 'pyq') {
    code = document.getElementById('course-code').value.trim();
    if (!code) return shakeField('course-code');
  } else if (docType === 'syllabus') {
    dept = document.getElementById('dept').value.trim();
    year = document.getElementById('year').value.trim();
    if (!dept) return shakeField('dept');
    if (!year) return shakeField('year');
  } else {
    return shakeField('doc-type');
  }

  if (!files || files.length === 0) {
    return shakeField('dropzone');
  }

  const btn = document.getElementById('upload-btn');
  btn.disabled = true; 
  document.getElementById('upload-ok').classList.remove('show');
  document.getElementById('upload-error').classList.remove('show');
  document.getElementById('prog-wrap').classList.add('show');

  try {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      btn.textContent = `Uploading ${i+1}/${files.length}…`;
      
      const stages = [[20,'Uploading PDF…'],[45,'Parsing document…'],[68,'Chunking content…'],[85,'Embedding chunks…'],[95,'Indexing into VectorDB…']];
      let si = 0;
      clearInterval(uploadTimer);
      uploadTimer = setInterval(() => { if (si < stages.length) { setProg(...stages[si]); si++; } }, 700);

      const formData = new FormData();
      formData.append('files', file); // changed 'file' to 'files' to match backend parameter
      formData.append('doc_type', docType);
      if (docType === 'pyq') formData.append('course_code', code);
      if (docType === 'syllabus') {
        // Use 'department' to match the FastAPI param name (backend also accepts 'dept' as alias)
        formData.append('department', dept);
        formData.append('year', year);
      }

      const res = await apiFetch('/upload', { method: 'POST', body: formData });
      clearInterval(uploadTimer);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed (HTTP ${res.status})`);
      }
      await res.json().catch(() => ({}));
    }

    setProg(100, 'Complete!');
    setTimeout(() => {
      document.getElementById('prog-wrap').classList.remove('show');
      document.getElementById('upload-ok').classList.add('show');
      btn.disabled = false; btn.innerHTML = 'Upload &amp; Index';
      setProg(0, '');
      fileInput._droppedFiles = null; fileInput.value = '';
      document.getElementById('dz-filename').textContent = '';
      fetchDocuments();
    }, 600);

  } catch (e) {
    clearInterval(uploadTimer);
    document.getElementById('prog-wrap').classList.remove('show');
    const errMsg = document.getElementById('upload-error-msg');
    errMsg.textContent = e.message === "Unauthorized" ? 'Unauthorized' : 'Network error or upload failed.';
    document.getElementById('upload-error').classList.add('show');
    btn.disabled = false; btn.innerHTML = 'Upload &amp; Index';
  }
}
window.doUpload = doUpload;

function shakeField(id) {
  const el = document.getElementById(id);
  el.style.borderColor = 'var(--brand)';
  setTimeout(() => el.style.borderColor = '', 1400);
}

export async function fetchDocuments() {
  const tbody = document.getElementById('documents-table-body');
  try {
    const res = await apiFetch('/documents');
    const docs = await res.json();

    if (!docs || docs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-4);padding:24px;">No documents indexed yet.</td></tr>';
      return;
    }

    tbody.innerHTML = docs.map(doc => `
      <tr>
        <td><strong>${doc.course_code || '—'}</strong></td>
        <td><span class="doc-type-badge">${doc.doc_type || '—'}</span></td>
        <td style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;">${doc.filename || '—'}</td>
        <td style="font-size:11.5px;color:var(--text-3);">${doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '—'}</td>
        <td><button class="btn-del" onclick="window.deleteDocument('${doc.id}')">Delete</button></td>
      </tr>`).join('');

    document.getElementById('stat-docs').textContent = docs.length;
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--brand);padding:20px;">Error loading documents.</td></tr>';
  }
}
window.fetchDocuments = fetchDocuments;

export async function deleteDocument(id) {
  if (!confirm('Delete this document and its indexed data?')) return;
  try {
    await apiFetch(`/documents/${id}`, { method: 'DELETE' });
    fetchDocuments();
  } catch (e) {
    alert('Failed to delete document or network error.');
  }
}
window.deleteDocument = deleteDocument;

let evalChartInstance = null;

async function loadEvalChart() {
  const canvas = document.getElementById('evalChart');
  if (!canvas || typeof window.Chart === 'undefined') return;

  let scores = [0.82, 0.91, 0.78, 0.88, 0.85];
  try {
    const res = await apiFetch('/evaluate');
    const data = await res.json();
    if (data.metrics) {
      scores = Object.values(data.metrics);
      const mean = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);
      document.getElementById('eval-mean').textContent = mean;
      document.getElementById('stat-mean').textContent = mean;
      document.getElementById('eval-count').textContent = data.num_samples || '5';
      document.getElementById('eval-status-text').textContent = `Last run: ${new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})} · Model: ${data.model || 'local'}`;
    }
  } catch (e) { /* use placeholder */ }

  if (evalChartInstance) evalChartInstance.destroy();

  evalChartInstance = new window.Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Context\nPrecision', 'Faithfulness', 'Answer\nRelevance', 'Context\nRecall', 'Answer\nCorrectness'],
      datasets: [{
        label: 'Score (0–1)',
        data: scores,
        backgroundColor: ['rgba(164,18,63,0.80)','rgba(164,18,63,0.70)','rgba(164,18,63,0.55)','rgba(164,18,63,0.70)','rgba(164,18,63,0.65)'],
        borderColor: 'rgba(164,18,63,0.90)',
        borderWidth: 1.5,
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1A1A1A', titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,.7)', padding: 10, cornerRadius: 8,
          callbacks: { label: ctx => ` Score: ${ctx.parsed.x.toFixed(2)}` }
        }
      },
      scales: {
        x: {
          min: 0, max: 1,
          grid: { color: 'rgba(0,0,0,.05)', drawTicks: false },
          border: { display: false },
          ticks: { color: '#8A8A8A', font: { family: "'Inter',sans-serif", size: 11 }, callback: v => v.toFixed(1) }
        },
        y: {
          grid: { display: false }, border: { display: false },
          ticks: { color: '#4A4A4A', font: { family: "'Inter',sans-serif", size: 11.5, weight: '500' } }
        }
      }
    }
  });
}

export async function runEvaluation() {
  const btn = document.getElementById('run-eval-btn');
  btn.disabled = true; btn.textContent = 'Running…';
  document.getElementById('eval-status-text').textContent = 'Running evaluation pipeline…';

  if (evalChartInstance) { evalChartInstance.destroy(); evalChartInstance = null; }
  await loadEvalChart();
  btn.disabled = false; btn.innerHTML = 'Run Evaluation Pipeline';
}
window.runEvaluation = runEvaluation;

export async function resetSystem() {
  const first = confirm('WARNING: This will permanently delete ALL documents, ALL knowledge graph data, and ALL uploaded files.\n\nThis cannot be undone. Are you absolutely sure?');
  if (!first) return;
  const second = confirm('Last chance — confirm you want to wipe the entire system.');
  if (!second) return;

  const statusEl = document.getElementById('reset-status');
  const btn = document.getElementById('reset-btn');
  statusEl.textContent = 'Resetting system… please wait.';
  btn.disabled = true;

  try {
    await apiFetch('/reset', { method: 'POST' });
    statusEl.style.color = 'var(--brand)';
    statusEl.textContent = 'System reset complete. All data has been wiped.';
    fetchDocuments();
  } catch (e) {
    statusEl.textContent = 'Network error or reset failed.';
  } finally {
    btn.disabled = false;
  }
}
window.resetSystem = resetSystem;
window.logout = logout;

window.addEventListener('DOMContentLoaded', () => {
  fetchDocuments();
  loadEvalChart();
});
