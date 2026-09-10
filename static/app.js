const stateJson = document.getElementById("state-json");
const timeline = document.getElementById("timeline");
const artifacts = document.getElementById("artifacts");
const runStatus = document.getElementById("run-status");
const healthEl = document.getElementById("health");
const samplesEl = document.getElementById("samples");
const runList = document.getElementById("run-list");
const approvalBox = document.getElementById("approval");
const approvalReason = document.getElementById("approval-reason");

let currentRun = null;
let eventSource = null;
let seenEvents = new Set();

async function j(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(status) {
  runStatus.textContent = status;
  runStatus.className = "pill " + (status || "idle");
}

function renderEvent(ev) {
  const id = String(ev.id);
  if (seenEvents.has(id)) return;
  seenEvents.add(id);
  const li = document.createElement("li");
  li.className = ev.kind === "tool_call" ? "tool" : ev.kind === "interrupt" ? "interrupt" : "";
  const when = (ev.ts || "").replace("T", " ").slice(11, 19);
  const node = ev.node ? ` · ${ev.node}` : "";
  const extra = ev.payload && ev.payload.tool ? ` ${ev.payload.tool}` : ev.payload && ev.payload.reason ? ` ${ev.payload.reason}` : "";
  li.innerHTML = `<span class="when">${when || "--:--:--"}</span><span class="kind">${ev.kind}${node}</span>${extra}`;
  timeline.appendChild(li);
  timeline.scrollTop = timeline.scrollHeight;
}

function renderState(run) {
  stateJson.textContent = JSON.stringify(run, null, 2);
  setStatus(run.status || "idle");
  artifacts.innerHTML = "";
  (run.artifacts || []).forEach((art) => {
    const div = document.createElement("div");
    div.className = "artifact";
    div.innerHTML = `<strong>${art.name}</strong> <span class="kind">${art.kind}</span><pre></pre>`;
    div.querySelector("pre").textContent = art.content;
    artifacts.appendChild(div);
  });
  if (run.status === "interrupted" && run.pending_action) {
    approvalBox.classList.remove("hidden");
    approvalReason.textContent = `${run.pending_action.action_type}: ${run.pending_action.reason}`;
  } else {
    approvalBox.classList.add("hidden");
  }
}

async function refreshRun(id) {
  const data = await j(`/v1/runs/${id}`);
  renderState(data.run);
  return data.run;
}

function subscribe(id) {
  if (eventSource) eventSource.close();
  seenEvents = new Set();
  timeline.innerHTML = "";
  eventSource = new EventSource(`/v1/runs/${id}/events`);
  eventSource.onmessage = (msg) => {
    try {
      renderEvent(JSON.parse(msg.data));
    } catch (_) {
      /* keepalive */
    }
  };
  ["node_start", "node_end", "tool_call", "interrupt", "resume", "run_start", "run_end", "error", "stream_end"].forEach((kind) => {
    eventSource.addEventListener(kind, (msg) => {
      try {
        renderEvent(JSON.parse(msg.data));
      } catch (_) {}
      if (kind === "interrupt" || kind === "run_end" || kind === "stream_end") {
        refreshRun(id);
        loadRuns();
      }
    });
  });
}

async function launch(task, mode) {
  const created = await j("/v1/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, mode }),
  });
  currentRun = created.run_id;
  setStatus(created.status);
  subscribe(created.run_id);
  const poll = async () => {
    const run = await refreshRun(created.run_id);
    if (run.status === "running" || run.status === "pending") {
      setTimeout(poll, 600);
    }
  };
  poll();
  loadRuns();
}

async function resume(approved) {
  if (!currentRun) return;
  const comment = document.getElementById("approval-comment").value;
  await j(`/v1/runs/${currentRun}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved, comment, actor: "mission-control" }),
  });
  subscribe(currentRun);
  const poll = async () => {
    const run = await refreshRun(currentRun);
    if (run.status === "running") setTimeout(poll, 600);
  };
  poll();
}

async function loadRuns() {
  const data = await j("/v1/runs");
  runList.innerHTML = "";
  (data.items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.status} · ${item.task.slice(0, 42)}`;
    if (item.run_id === currentRun) li.classList.add("active");
    li.onclick = async () => {
      currentRun = item.run_id;
      await refreshRun(item.run_id);
      const hist = await j(`/v1/runs/${item.run_id}/events/history`);
      seenEvents = new Set();
      timeline.innerHTML = "";
      (hist.items || []).forEach(renderEvent);
      loadRuns();
    };
    runList.appendChild(li);
  });
}

document.getElementById("run-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  launch(document.getElementById("task").value, document.getElementById("mode").value);
});
document.getElementById("task").addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
    ev.preventDefault();
    document.getElementById("run-form").requestSubmit();
  }
});
document.getElementById("copy-state").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(stateJson.textContent);
    document.getElementById("copy-state").textContent = "Copied";
    setTimeout(() => { document.getElementById("copy-state").textContent = "Copy JSON"; }, 1200);
  } catch {
    /* ignore */
  }
});
document.getElementById("approve").onclick = () => resume(true);
document.getElementById("reject").onclick = () => resume(false);

(async function init() {
  try {
    const h = await j("/v1/health");
    healthEl.textContent = `ok · ${h.llm} · ${h.documents} docs`;
    const meta = await j("/v1/meta");
    (meta.samples || []).forEach((sample) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = sample.title;
      btn.onclick = () => {
        document.getElementById("task").value = sample.task;
        document.getElementById("mode").value = sample.mode;
      };
      samplesEl.appendChild(btn);
    });
    await loadRuns();
  } catch (err) {
    healthEl.textContent = "offline";
    healthEl.style.color = "var(--danger)";
  }
})();
