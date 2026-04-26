'use strict';

// ── Feature definitions ────────────────────────────────────────────────────

const FIELD_GROUPS = [
  { id:'transactions', label:'Transactions', icon:'💳', fields:[
    { key:'n_transactions',      label:'Total Transactions',  type:'number', min:0, step:1    },
    { key:'n_cancels',           label:'Cancellations',       type:'number', min:0, step:1    },
    { key:'ever_canceled',       label:'Ever Canceled',       type:'binary'                   },
    { key:'avg_discount_pct',    label:'Avg Discount %',      type:'number', min:0, max:1, step:.01 },
    { key:'avg_plan_days',       label:'Avg Plan Days',       type:'number', min:0            },
    { key:'avg_price',           label:'Avg Price',           type:'number', min:0            },
    { key:'n_unique_plans',      label:'Unique Plans',        type:'number', min:0, step:1    },
    { key:'n_payment_methods',   label:'Payment Methods',     type:'number', min:0, step:1    },
    { key:'last_is_cancel',      label:'Last: Cancel',        type:'binary'                   },
    { key:'last_is_auto_renew',  label:'Last: Auto Renew',    type:'binary'                   },
    { key:'last_plan_days',      label:'Last Plan Days',      type:'number', min:0            },
    { key:'last_price',          label:'Last Price',          type:'number', min:0            },
    { key:'last_list_price',     label:'Last List Price',     type:'number', min:0            },
    { key:'price_trend',         label:'Price Trend',         type:'number', step:.01         },
    { key:'last_payment_method', label:'Payment Method ID',   type:'number', min:0, step:1   },
  ]},
  { id:'membership', label:'Membership', icon:'👤', fields:[
    { key:'city',               label:'City',               type:'number', min:1, step:1     },
    { key:'registered_via',     label:'Reg. Channel',       type:'number', min:1, step:1     },
    { key:'gender_enc',         label:'Gender',             type:'select',
      options:[{v:0,l:'Unknown'},{v:1,l:'Male'},{v:2,l:'Female'}]                            },
    { key:'age',                label:'Age',                type:'number', min:0, max:100     },
    { key:'bd_valid',           label:'Valid Birthday',     type:'binary'                    },
    { key:'tenure_days',        label:'Tenure (days)',      type:'number', min:0             },
    { key:'has_member_record',  label:'Has Member Record',  type:'binary'                    },
  ]},
  { id:'listening', label:'Listening', icon:'🎵', fields:[
    { key:'n_days',               label:'Active Days',          type:'number', min:0          },
    { key:'avg_daily_secs',       label:'Avg Daily Seconds',    type:'number', min:0          },
    { key:'avg_daily_completed',  label:'Avg Daily Completed',  type:'number', min:0          },
    { key:'avg_daily_unq',        label:'Avg Daily Unique',     type:'number', min:0          },
    { key:'completion_ratio',     label:'Completion Ratio',     type:'number', min:0, max:1, step:.01 },
    { key:'days_since_last',      label:'Days Since Last',      type:'number', min:0          },
    { key:'listening_trend',      label:'Listening Trend',      type:'number', step:.01       },
    { key:'has_log_record',       label:'Has Log Record',       type:'binary'                 },
  ]},
  { id:'expiry', label:'Expiry & Renewals', icon:'📅', fields:[
    { key:'days_until_expire',    label:'Days Until Expire',    type:'number'                 },
    { key:'is_expired',           label:'Is Expired',           type:'binary'                 },
    { key:'auto_renew_at_expire', label:'Auto Renew',           type:'binary'                 },
    { key:'cancel_at_expire',     label:'Cancel at Expire',     type:'binary'                 },
    { key:'n_renewals',           label:'Renewals',             type:'number', min:0          },
    { key:'prev_churn',           label:'Previous Churn',       type:'binary'                 },
    { key:'cancel_before_expire', label:'Cancel Before Expire', type:'binary'                 },
  ]},
  { id:'recency', label:'Recency', icon:'⏱️', fields:[
    { key:'days_since_last_tx', label:'Days Since Last Tx', type:'number', min:0             },
    { key:'had_tx_last_7d',     label:'Tx Last 7d',         type:'binary'                    },
    { key:'had_tx_last_30d',    label:'Tx Last 30d',        type:'binary'                    },
    { key:'n_tx_last_30d',      label:'Tx Count (30d)',     type:'number', min:0             },
  ]},
  { id:'multiwindow', label:'Multi-window Listening', icon:'📊', fields:[
    { key:'n_days_7d',        label:'Active Days (7d)',   type:'number', min:0               },
    { key:'secs_per_day_7d',  label:'Secs/Day (7d)',      type:'number', min:0               },
    { key:'n_days_90d',       label:'Active Days (90d)',  type:'number', min:0               },
    { key:'secs_per_day_90d', label:'Secs/Day (90d)',     type:'number', min:0               },
    { key:'trend_7d',         label:'Trend (7d)',         type:'number', step:1              },
    { key:'trend_7d_vs_30d',  label:'Trend 7d vs 30d',   type:'number', step:1              },
  ]},
];

// ── Presets ───────────────────────────────────────────────────────────────

const BASE = {
  n_transactions:12, n_cancels:1, ever_canceled:1, avg_discount_pct:.05,
  avg_plan_days:30, avg_price:149, n_unique_plans:2, n_payment_methods:1,
  last_is_cancel:0, last_is_auto_renew:1, last_plan_days:30, last_price:149,
  last_list_price:149, price_trend:0, last_payment_method:36,
  city:1, registered_via:4, gender_enc:0, age:28, bd_valid:1,
  tenure_days:365, has_member_record:1,
  n_days:20, avg_daily_secs:3600, avg_daily_completed:10, avg_daily_unq:8,
  completion_ratio:.75, days_since_last:2, listening_trend:.1, has_log_record:1,
  days_until_expire:15, is_expired:0, auto_renew_at_expire:1, cancel_at_expire:0,
  n_renewals:11, prev_churn:0,
  days_since_last_tx:5, had_tx_last_7d:1, had_tx_last_30d:1, n_tx_last_30d:1,
  cancel_before_expire:0,
  n_days_7d:5, secs_per_day_7d:3200, n_days_90d:60, secs_per_day_90d:3400,
  trend_7d:-100, trend_7d_vs_30d:-50,
};

const PRESETS = {
  active: { msno:'demo_active_user', ...BASE },
  expired: {
    msno:'demo_expired_user', ...BASE,
    days_until_expire:-30, is_expired:1, auto_renew_at_expire:0,
    cancel_at_expire:1, last_is_cancel:1, cancel_before_expire:1,
    prev_churn:1, days_since_last:45,
    n_days:2, avg_daily_secs:400, completion_ratio:.08,
    listening_trend:-.5, n_days_7d:0, secs_per_day_7d:0,
  },
  new_user: {
    msno:'demo_new_user', ...BASE,
    n_transactions:1, n_cancels:0, ever_canceled:0,
    tenure_days:15, n_renewals:0, prev_churn:0,
    n_days_90d:5, secs_per_day_90d:2000,
  },
};

// ── State ──────────────────────────────────────────────────────────────────

let gaugeProb   = 0;
let batchUsers  = [];
let batchResult = null;

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const savedKey = localStorage.getItem('churn_api_key') || '';
  document.getElementById('api-key').value = savedKey;
  document.getElementById('api-key').addEventListener('change', e => {
    localStorage.setItem('churn_api_key', e.target.value);
  });

  buildForm();
  applyPreset('active');
  fetchModelBadge();
});

// ── Navigation ─────────────────────────────────────────────────────────────

function navigate(view) {
  document.querySelectorAll('.view').forEach(v => v.style.display = 'none');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`view-${view}`).style.display = '';
  document.querySelector(`[data-view="${view}"]`).classList.add('active');
  if (view === 'dashboard') loadDashboard();
}

// ── Form builder ───────────────────────────────────────────────────────────

function buildForm() {
  const container = document.getElementById('feature-groups');
  container.innerHTML = FIELD_GROUPS.map((g, i) => `
    <div class="acc-item ${i === 0 ? 'open' : ''}" id="acc-${g.id}">
      <button class="acc-header" onclick="toggleAcc('${g.id}')">
        <span class="acc-icon">
          <span>${g.icon}</span>
          <span>${g.label}</span>
        </span>
        <svg class="acc-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </button>
      <div class="acc-body">
        <div class="field-grid">
          ${g.fields.map(f => renderField(f)).join('')}
        </div>
      </div>
    </div>
  `).join('');
}

function renderField(f) {
  if (f.type === 'binary') {
    return `<div class="field">
      <label class="field-label">${f.label}</label>
      <select class="field-input" id="f-${f.key}">
        <option value="0">No — 0</option>
        <option value="1">Yes — 1</option>
      </select>
    </div>`;
  }
  if (f.type === 'select') {
    const opts = f.options.map(o => `<option value="${o.v}">${o.l}</option>`).join('');
    return `<div class="field">
      <label class="field-label">${f.label}</label>
      <select class="field-input" id="f-${f.key}">${opts}</select>
    </div>`;
  }
  const attrs = [
    f.min  !== undefined ? `min="${f.min}"` : '',
    f.max  !== undefined ? `max="${f.max}"` : '',
    f.step !== undefined ? `step="${f.step}"` : 'step="any"',
  ].join(' ');
  return `<div class="field">
    <label class="field-label">${f.label}</label>
    <input class="field-input" type="number" id="f-${f.key}" ${attrs} value="0">
  </div>`;
}

function toggleAcc(id) {
  document.getElementById(`acc-${id}`).classList.toggle('open');
}

// ── Presets ────────────────────────────────────────────────────────────────

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  document.getElementById('f-msno').value = p.msno;
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById(`f-${k}`);
    if (el) el.value = v;
  }
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`[data-preset="${name}"]`)?.classList.add('active');
}

function collectFormData() {
  const msno = document.getElementById('f-msno').value.trim() || 'user_unknown';
  const data = { msno };
  for (const g of FIELD_GROUPS)
    for (const f of g.fields) {
      const el = document.getElementById(`f-${f.key}`);
      data[f.key] = el ? parseFloat(el.value) : 0;
    }
  return data;
}

// ── Predict ────────────────────────────────────────────────────────────────

async function submitPredict() {
  const btn  = document.getElementById('predict-btn');
  const data = collectFormData();

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  try {
    const res = await apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ features: data }),
    });
    if (res.status === 401) { toast('Invalid API key', 'error'); return; }
    if (res.status === 422) { toast('Validation error — check feature values', 'error'); return; }
    if (!res.ok)            { toast('Prediction failed', 'error'); return; }

    showResult(await res.json());
    toast('Prediction complete', 'success');

  } catch { toast('Connection error', 'error'); }
  finally {
    btn.disabled = false;
    btn.innerHTML = 'Predict Churn <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';
  }
}

function showResult(r) {
  document.getElementById('result-placeholder').style.display = 'none';
  const card = document.getElementById('result-card');
  card.style.display = 'flex';

  document.getElementById('res-msno').textContent  = r.msno;
  document.getElementById('res-prob').textContent  = `${(r.churn_prob * 100).toFixed(2)}%`;
  document.getElementById('res-model').textContent = r.model_version;
  document.getElementById('res-time').textContent  = fmtTime(r.predicted_at);

  const badge = document.getElementById('churn-badge');
  if (r.churn_label === 1) {
    badge.textContent  = '⚠ Churn Risk';
    badge.className    = 'churn-badge danger';
  } else {
    badge.textContent  = '✓ Low Risk';
    badge.className    = 'churn-badge success';
  }

  animateGauge(r.churn_prob);
}

// ── Gauge ──────────────────────────────────────────────────────────────────

function animateGauge(target, duration = 900) {
  const start     = gaugeProb;
  const t0        = performance.now();

  function frame(now) {
    const elapsed = now - t0;
    const t       = Math.min(elapsed / duration, 1);
    const eased   = 1 - Math.pow(1 - t, 3);
    const current = start + (target - start) * eased;
    setGauge(current);
    if (t < 1) requestAnimationFrame(frame);
    else gaugeProb = target;
  }
  requestAnimationFrame(frame);
}

function setGauge(prob) {
  const deg   = Math.round(prob * 360);
  const pct   = Math.round(prob * 100);
  const color = prob < .3 ? '#10b981' : prob < .5 ? '#f59e0b' : '#ef4444';
  const gauge = document.getElementById('gauge');

  gauge.style.background  = `conic-gradient(${color} ${deg}deg, var(--surface-3) 0deg)`;
  gauge.style.boxShadow   = `0 0 28px ${color}40`;

  const pctEl = document.getElementById('gauge-pct');
  pctEl.textContent = `${pct}%`;
  pctEl.style.color = color;
}

// ── Batch ──────────────────────────────────────────────────────────────────

function dragOver(e)  { e.preventDefault(); document.getElementById('dropzone').classList.add('drag-over'); }
function dragLeave()  { document.getElementById('dropzone').classList.remove('drag-over'); }
function dropFile(e)  { e.preventDefault(); dragLeave(); readFile(e.dataTransfer.files[0]); }
function fileSelected(e) { readFile(e.target.files[0]); }

function readFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.csv')) { toast('Please upload a .csv file', 'error'); return; }
  const reader = new FileReader();
  reader.onload = e => {
    try {
      batchUsers = parseCSV(e.target.result);
      if (!batchUsers.length) { toast('No valid rows found in CSV', 'error'); return; }
      document.getElementById('batch-info').textContent =
        `📂 ${file.name}  ·  ${batchUsers.length.toLocaleString()} users ready`;
      document.getElementById('batch-preview').style.display = '';
      toast(`Loaded ${batchUsers.length} users`, 'info');
    } catch(err) { toast('CSV parse error: ' + err.message, 'error'); }
  };
  reader.readAsText(file);
}

function parseCSV(text) {
  const lines   = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  return lines.slice(1)
    .filter(l => l.trim())
    .map(line => {
      const vals = line.split(',').map(v => v.trim().replace(/"/g, ''));
      const row  = {};
      headers.forEach((h, i) => {
        const v = vals[i] ?? '';
        row[h]  = isNaN(v) || v === '' ? v : parseFloat(v);
      });
      return row;
    })
    .filter(r => r.msno !== undefined);
}

async function submitBatch() {
  if (!batchUsers.length) return;
  const btn = document.getElementById('batch-submit-btn');
  btn.disabled = true;

  const prog = document.getElementById('batch-progress');
  const fill = document.getElementById('progress-fill');
  const txt  = document.getElementById('progress-text');
  prog.style.display = '';

  const anim = setInterval(() => {
    const w = parseFloat(fill.style.width) || 0;
    if (w < 85) fill.style.width = (w + 3) + '%';
  }, 100);

  txt.textContent = `Processing ${batchUsers.length.toLocaleString()} users…`;

  try {
    const res = await apiFetch('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ users: batchUsers }),
    });
    clearInterval(anim);
    fill.style.width = '100%';

    if (!res.ok) { toast('Batch failed: ' + (await res.text()), 'error'); return; }

    batchResult = await res.json();
    renderBatchResults(batchResult);
    toast(`${batchResult.total} predictions complete`, 'success');

  } catch { toast('Connection error', 'error'); }
  finally {
    btn.disabled = false;
    setTimeout(() => { prog.style.display = 'none'; fill.style.width = '0%'; }, 800);
  }
}

function renderBatchResults(data) {
  const preds = [...data.predictions].sort((a, b) => b.churn_prob - a.churn_prob);
  document.getElementById('batch-result-title').textContent =
    `${data.total.toLocaleString()} predictions  ·  ${preds.filter(p => p.churn_label).length} at risk`;

  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = preds.slice(0, 200).map(p => {
    const pct   = (p.churn_prob * 100).toFixed(1);
    const cls   = p.churn_prob < .3 ? 'low' : p.churn_prob < .5 ? 'medium' : 'high';
    const label = p.churn_label ? '<span class="label-badge churn">Churn</span>'
                                : '<span class="label-badge no-churn">Safe</span>';
    return `<tr>
      <td>${p.msno}</td>
      <td>
        <div class="prob-cell">
          <div class="prob-track"><div class="prob-fill ${cls}" style="width:${pct}%"></div></div>
          <span class="prob-text">${pct}%</span>
        </div>
      </td>
      <td>${label}</td>
      <td style="color:var(--text-3)">${p.model_version ?? '—'}</td>
    </tr>`;
  }).join('');

  document.getElementById('batch-results').style.display = '';
}

function downloadResults() {
  if (!batchResult) return;
  const rows = [['msno','churn_prob','churn_label','model_version']];
  batchResult.predictions.forEach(p =>
    rows.push([p.msno, p.churn_prob, p.churn_label, p.model_version ?? ''])
  );
  downloadCSV(rows, 'churn_predictions.csv');
}

function downloadTemplate() {
  const allFields = FIELD_GROUPS.flatMap(g => g.fields.map(f => f.key));
  const headers   = ['msno', ...allFields];
  const example   = [PRESETS.active.msno, ...allFields.map(k => PRESETS.active[k] ?? 0)];
  downloadCSV([headers, example], 'churn_template.csv');
  toast('Template downloaded', 'info');
}

// ── Dashboard ──────────────────────────────────────────────────────────────

async function loadDashboard() {
  document.getElementById('stat-total').textContent      = '…';
  document.getElementById('stat-churn-rate').textContent = '…';
  document.getElementById('stat-avg-prob').textContent   = '…';
  document.getElementById('stat-model').textContent      = '…';

  try {
    const [healthRes, recentRes] = await Promise.all([
      fetch('/health'),
      apiFetch('/predictions/recent?limit=300'),
    ]);

    const health = await healthRes.json();
    const recent = recentRes.ok ? await recentRes.json() : { predictions: [] };
    const preds  = recent.predictions || [];

    renderStats(health, preds);
    renderChart(preds);
    renderRecentTable(preds.slice(0, 25));

  } catch { toast('Could not load dashboard data', 'error'); }
}

function renderStats(health, preds) {
  const todayStr = new Date().toDateString();
  const today    = preds.filter(p => new Date(p.predicted_at).toDateString() === todayStr);
  const rate     = today.length ? today.filter(p => p.churn_label).length / today.length : 0;
  const avg      = today.length ? today.reduce((s, p) => s + p.churn_prob, 0) / today.length : 0;

  document.getElementById('stat-total').textContent      = today.length.toLocaleString();
  document.getElementById('stat-churn-rate').textContent = `${(rate * 100).toFixed(1)}%`;
  document.getElementById('stat-avg-prob').textContent   = `${(avg  * 100).toFixed(1)}%`;
  document.getElementById('stat-model').textContent      = health.model_version || '—';
}

function renderChart(preds) {
  const chart = document.getElementById('daily-chart');
  if (!preds.length) { chart.innerHTML = '<div class="chart-empty">No predictions yet</div>'; return; }

  const days = {};
  for (let i = 6; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    days[d.toDateString()] = { count: 0, churn: 0 };
  }
  preds.forEach(p => {
    const k = new Date(p.predicted_at).toDateString();
    if (days[k]) { days[k].count++; if (p.churn_label) days[k].churn++; }
  });

  const entries = Object.entries(days);
  const maxRate = Math.max(.01, ...entries.map(([,v]) => v.count ? v.churn / v.count : 0));

  chart.innerHTML = entries.map(([day, v]) => {
    const rate = v.count ? v.churn / v.count : 0;
    const h    = Math.round((rate / maxRate) * 100);
    const cls  = rate > .5 ? 'high' : rate > .25 ? 'medium' : 'low';
    const lbl  = new Date(day).toLocaleDateString('en', { weekday:'short' });
    return `<div class="chart-bar-wrap">
      <div class="chart-bar ${cls}" style="height:${h}%" title="${(rate*100).toFixed(1)}%"></div>
      <span class="chart-label">${lbl}</span>
    </div>`;
  }).join('');
}

function renderRecentTable(preds) {
  document.getElementById('recent-count').textContent =
    preds.length ? `${preds.length} most recent` : '';
  const tbody = document.getElementById('recent-tbody');
  if (!preds.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No predictions yet</td></tr>';
    return;
  }
  tbody.innerHTML = preds.map(p => {
    const pct   = (p.churn_prob * 100).toFixed(1);
    const cls   = p.churn_prob < .3 ? 'low' : p.churn_prob < .5 ? 'medium' : 'high';
    const label = p.churn_label
      ? '<span class="label-badge churn">Churn</span>'
      : '<span class="label-badge no-churn">Safe</span>';
    return `<tr>
      <td style="font-weight:500;color:var(--text)">${p.msno}</td>
      <td>
        <div class="prob-cell">
          <div class="prob-track"><div class="prob-fill ${cls}" style="width:${pct}%"></div></div>
          <span class="prob-text">${pct}%</span>
        </div>
      </td>
      <td>${label}</td>
      <td style="color:var(--text-3)">${p.source ?? 'api'}</td>
      <td style="color:var(--text-3)">${fmtTime(p.predicted_at)}</td>
    </tr>`;
  }).join('');
}

// ── Model badge ────────────────────────────────────────────────────────────

async function fetchModelBadge() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    document.getElementById('model-version-text').textContent =
      data.model_version ? `v${data.model_version}` : 'Unknown';
    document.querySelector('.badge-dot').style.background =
      data.model_loaded ? 'var(--success)' : 'var(--danger)';
  } catch {
    document.getElementById('model-version-text').textContent = 'Offline';
    document.querySelector('.badge-dot').style.background = 'var(--danger)';
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────

function getApiKey() {
  return document.getElementById('api-key').value.trim()
      || localStorage.getItem('churn_api_key') || '';
}

function apiFetch(url, options = {}) {
  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': getApiKey(),
      ...(options.headers || {}),
    },
  });
}

function toast(msg, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('en', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function downloadCSV(rows, filename) {
  const csv  = rows.map(r => r.map(c => `"${c}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type:'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}
