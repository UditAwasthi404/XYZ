const $ = (s) => document.querySelector(s);
let tab = "open";
const LABEL = { OUT_OF_STOCK: "Out of stock", LOW_STOCK: "Low stock", MISPLACED: "Misplaced item" };

function fmtDur(s) {
  if (s == null) return "–";
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
function toast(msg) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t); setTimeout(() => t.remove(), 2600);
}

async function loadMetrics() {
  const m = await (await fetch("/metrics")).json();
  $("#m-osa").textContent = m.on_shelf_availability_pct == null ? "–" : m.on_shelf_availability_pct + "%";
  $("#m-comp").textContent = m.planogram_compliance_pct == null ? "–" : m.planogram_compliance_pct + "%";
  $("#m-mttr").textContent = fmtDur(m.mean_time_to_resolve_s);
  $("#m-open").textContent = m.open_events;
  $("#m-res").textContent = `${m.resolved_events} resolved`;
}

function alertItem(e) {
  const li = document.createElement("li");
  const resolved = e.status === "resolved";
  li.className = resolved ? "resolved" : e.priority;
  const what = e.type === "MISPLACED" ? `${e.sku} in ${e.details.expected_sku} slot`
    : e.type === "LOW_STOCK" ? `${e.sku} · ${e.details.facings}/${e.details.expected} facings` : e.sku;
  const when = new Date((resolved ? e.resolved_at : e.created_at) * 1000).toLocaleTimeString();
  li.innerHTML = `
    <div class="al-main">
      <div class="al-title"><span class="badge ${resolved ? "ok" : e.priority}">${resolved ? "RESOLVED" : e.priority}</span>${LABEL[e.type]}</div>
      <div>${what}</div>
      <div class="al-meta">Aisle ${e.aisle} · Bay ${e.shelf_id} · Shelf ${e.shelf ?? "-"} · ${when}${e.status === "acked" ? " · acknowledged" : ""}</div>
    </div>
    <div class="al-actions">
      <a href="${e.evidence}" target="_blank">photo proof</a>
      ${e.status === "open" ? `<button data-ack="${e.id}">Ack</button>` : ""}
    </div>`;
  return li;
}

async function loadEvents() {
  const evs = await (await fetch(`/events?status=${tab}`)).json();
  const ul = $("#alerts"); ul.innerHTML = "";
  if (!evs.length) ul.innerHTML = `<p class="empty">${tab === "open" ? "No open tasks. Shelves look good." : "Nothing resolved yet."}</p>`;
  evs.forEach((e) => ul.appendChild(alertItem(e)));
}

async function runAudit(blob, name, shelfId) {
  const btn = $("#form button"); btn.disabled = true;
  const fd = new FormData(); fd.append("image", blob, name); fd.append("shelf_id", shelfId);
  try {
    const r = await fetch("/audit", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    $("#viewer").innerHTML = `<img src="${j.evidence}?t=${Date.now()}" alt="annotated shelf" />`;
    $("#summary").innerHTML = `Bay <b>${j.shelf_id}</b> · detector <b>${j.backend}</b> · <b>${j.products}</b> products · <b>${j.gaps}</b> gaps · <b>${j.events_detected}</b> issues · <b>${j.created.length}</b> new tasks · <b>${j.auto_resolved.length}</b> auto-closed${j.people_blurred ? ` · ${j.people_blurred} people blurred` : ""}`;
    if (j.auto_resolved.length) toast(`${j.auto_resolved.length} task(s) auto-closed by this image`);
  } catch (err) { toast("Audit failed: " + err.message); }
  btn.disabled = false; loadMetrics(); loadEvents();
}

async function init() {
  const shelves = await (await fetch("/shelves")).json();
  $("#shelf").innerHTML = shelves.map((s) => `<option>${s.shelf_id}</option>`).join("") + `<option value="CAM-NEW">CAM-NEW (no planogram)</option>`;
  const samples = await (await fetch("/samples")).json();
  $("#samples").innerHTML = samples.map((s) => `<button data-img="${s.image}" data-shelf="${s.shelf_id}">${s.image.replace(".png", "")}</button>`).join(" ");
  $("#samples").addEventListener("click", async (ev) => {
    const b = ev.target.closest("button"); if (!b) return;
    const blob = await (await fetch(`/sample/${b.dataset.img}`)).blob();
    $("#shelf").value = b.dataset.shelf;
    runAudit(blob, b.dataset.img, b.dataset.shelf);
  });
  $("#form").addEventListener("submit", (ev) => {
    ev.preventDefault(); const f = $("#file").files[0];
    if (!f) return toast("Choose an image first");
    runAudit(f, f.name, $("#shelf").value);
  });
  $("#alerts").addEventListener("click", async (ev) => {
    const id = ev.target.dataset.ack; if (!id) return;
    await fetch(`/events/${id}/ack`, { method: "POST" }); loadEvents(); loadMetrics();
  });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on"); tab = b.dataset.s; loadEvents();
  }));

  const es = new EventSource("/events/stream");
  es.onopen = () => { $("#dot").classList.add("on"); $("#liveText").textContent = "live"; };
  es.onerror = () => { $("#dot").classList.remove("on"); $("#liveText").textContent = "reconnecting…"; };
  es.onmessage = (m) => {
    const { kind, event } = JSON.parse(m.data);
    if (kind === "created" && event.priority === "HIGH") toast(`New HIGH task: ${LABEL[event.type]} ${event.sku}`);
    clearTimeout(window._rt); window._rt = setTimeout(() => { loadEvents(); loadMetrics(); }, 150);
  };
  const latest = await (await fetch("/audits/latest")).json();
  if (latest.length) {
    const a = latest[0];
    $("#viewer").innerHTML = `<img src="${a.evidence}" alt="annotated shelf" />`;
    $("#summary").innerHTML = `Last audit: bay <b>${a.shelf_id}</b> · detector <b>${a.backend}</b> · <b>${a.n_products}</b> products · <b>${a.n_gaps}</b> gaps`;
  }
  loadMetrics(); loadEvents();
}
init();
