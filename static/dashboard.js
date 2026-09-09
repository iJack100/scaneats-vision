const POLL_MS = 2000;

function fmtSeconds(s) {
  if (s === null || s === undefined) return "-";
  const total = Math.round(s);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  return `${m}m ${sec.toString().padStart(2, "0")}s`;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

async function refreshStatus() {
  const s = await getJSON("/api/status");
  const el = document.getElementById("status-badges");
  el.innerHTML = `
    <span>${s.running ? "En vivo" : "Detenido"}</span>
    <span>${s.frames_processed} fotogramas procesados</span>
    <span>${s.tables_configured} mesas configuradas</span>
    <span>Alerta &gt; ${Math.round(s.alert_threshold_seconds / 60)} min</span>
  `;
}

async function refreshTables() {
  const tables = await getJSON("/api/tables");
  const grid = document.getElementById("tables-grid");
  grid.innerHTML = tables.map(t => `
    <div class="table-card ${t.status}">
      <div class="name">${t.name}</div>
      <div class="status">${t.status}</div>
      <div class="timer">${t.waiting_seconds !== null ? fmtSeconds(t.waiting_seconds) : "&mdash;"}</div>
      ${t.last_wait_seconds !== null ? `<div class="hint">Ultima espera: ${fmtSeconds(t.last_wait_seconds)}</div>` : ""}
    </div>
  `).join("") || `<p class="hint">Sin mesas configuradas.</p>`;
}

async function refreshWaiters() {
  const waiters = await getJSON("/api/waiters");
  const tbody = document.querySelector("#waiters-table tbody");
  if (!waiters.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="4">Aun no se confirma ningun mesero.</td></tr>`;
    return;
  }
  tbody.innerHTML = waiters.map(w => `
    <tr>
      <td>#${w.waiter_track_id}</td>
      <td>${w.interacciones}</td>
      <td>${w.mesas_atendidas}</td>
      <td class="${w.en_pantalla ? "live" : ""}">${w.en_pantalla ? "En pantalla" : "Fuera de camara"}</td>
    </tr>
  `).join("");
}

async function refreshHeatmap() {
  const rows = await getJSON("/api/heatmap");
  const max = Math.max(1, ...rows.map(r => r.occupied_seconds));
  const el = document.getElementById("heatmap");
  el.innerHTML = rows.map(r => `
    <div class="heatmap-row">
      <span>${r.name}</span>
      <span class="heatmap-bar-bg"><span class="heatmap-bar-fill" style="width:${(r.occupied_seconds / max * 100).toFixed(0)}%"></span></span>
      <span>${r.times_occupied}x</span>
    </div>
  `).join("") || `<p class="hint">Sin datos de ocupacion todavia.</p>`;
}

async function refreshAlerts() {
  const alerts = await getJSON("/api/alerts");
  const list = document.getElementById("alerts-list");
  if (!alerts.length) {
    list.innerHTML = `<li class="empty">Sin alertas registradas.</li>`;
    return;
  }
  list.innerHTML = alerts.map(a => `
    <li class="${a.active ? "" : "resolved"}">
      <strong>${a.table_name}</strong> · espera ${fmtSeconds(a.wait_seconds_at_trigger)}
      ${a.active ? "&mdash; ACTIVA" : "&mdash; resuelta"}
    </li>
  `).join("");
}

async function tick() {
  try {
    await Promise.all([refreshStatus(), refreshTables(), refreshWaiters(), refreshHeatmap(), refreshAlerts()]);
  } catch (err) {
    console.error("ScanEats dashboard poll error:", err);
  }
}

tick();
setInterval(tick, POLL_MS);
