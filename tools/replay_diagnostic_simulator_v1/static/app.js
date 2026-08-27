"use strict";

const state = {
  meta: null,
  timeline: [],
  frame: null,
  step: 0,
  seat: 0,
  playing: false,
  timer: null,
  frameCache: new Map(),
  requestId: 0,
  viewerReady: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const viewer = $("#official-viewer");
const slider = $("#step-slider");
const money = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 });

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => { node.className = "toast"; }, 2600);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setBusy(busy) {
  $("#load-path").disabled = busy;
  $("#upload-file").disabled = busy;
  $("#load-path").textContent = busy ? "解析中…" : "载入路径";
}

async function activateReplay(meta) {
  stopPlayback();
  state.meta = meta;
  state.timeline = [];
  state.frameCache.clear();
  state.step = 0;
  state.viewerReady = false;
  renderMeta();
  slider.max = String(Math.max(0, meta.steps - 1));
  slider.value = "0";
  slider.disabled = false;
  $("#viewer-empty").hidden = true;
  viewer.hidden = false;
  viewer.src = `/official-viewer?generation=${encodeURIComponent(meta.generation)}&v=${Date.now()}`;
  const timelineResponse = await api("/api/timeline");
  state.timeline = timelineResponse.timeline;
  await setStep(0, true);
  toast(`已载入 Episode ${meta.episode_id}`);
}

async function loadPath() {
  const path = $("#replay-path").value.trim();
  if (!path) return toast("请先粘贴 Replay JSON 路径", true);
  setBusy(true);
  try {
    const data = await api("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    await activateReplay(data.meta);
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function uploadFile(file) {
  if (!file) return;
  setBusy(true);
  try {
    const data = await api(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: file,
    });
    $("#replay-path").value = file.name;
    await activateReplay(data.meta);
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    $("#upload-file").value = "";
  }
}

function renderMeta() {
  const meta = state.meta;
  if (!meta) return;
  $("#meta-strip").classList.remove("is-empty");
  $("#episode-id").textContent = meta.episode_id ?? "—";
  $("#players").textContent = `${meta.players[0]} vs ${meta.players[1]}`;
  $("#rewards").textContent = (meta.rewards || []).map(v => money.format(v)).join(" : ") || "—";
  $("#version").textContent = meta.module_version || meta.version || "—";
  $("#source-path").textContent = meta.path || "—";
  $("#source-path").title = meta.path || "";
  $$(".seat-tab").forEach((button, seat) => {
    button.textContent = `${seat} · ${meta.players[seat]}`;
    button.title = meta.players[seat];
  });
}

async function fetchFrame(step) {
  if (state.frameCache.has(step)) return state.frameCache.get(step);
  const data = await api(`/api/frame?step=${step}`);
  state.frameCache.set(step, data.frame);
  if (state.frameCache.size > 60) state.frameCache.delete(state.frameCache.keys().next().value);
  return data.frame;
}

async function setStep(rawStep, force = false) {
  if (!state.meta) return;
  const step = Math.max(0, Math.min(Number(rawStep), state.meta.steps - 1));
  if (!force && step === state.step && state.frame) return;
  state.step = step;
  slider.value = String(step);
  updateStepReadout();
  if (state.viewerReady && viewer.contentWindow) {
    viewer.contentWindow.postMessage({ step, playing: false, parentHandlesUi: true }, "*");
  }
  const requestId = ++state.requestId;
  try {
    const frame = await fetchFrame(step);
    if (requestId !== state.requestId) return;
    state.frame = frame;
    renderFrame();
    requestAnimationFrame(applyViewerOverlays);
  } catch (error) {
    toast(error.message, true);
  }
}

function updateStepReadout() {
  const step = state.step;
  const day = state.frame?.step === step ? state.frame.day : Math.floor(step / 24);
  const hour = state.frame?.step === step ? state.frame.hour : step % 24;
  $("#day-hour").textContent = `第 ${day + 1} 天 · 第 ${hour + 1} 回合`;
  $("#step-number").textContent = `Frame ${step} / ${Math.max(0, (state.meta?.steps || 1) - 1)}`;
}

function severityCount(alarms) {
  return alarms.reduce((total, alarm) => total + Number(alarm.count || 0), 0);
}

function renderFrame() {
  if (!state.frame || !state.meta) return;
  updateStepReadout();
  const seat = state.frame.seats[state.seat];
  $("#diagnostic-title").textContent = `${state.meta.players[state.seat]} · Frame ${state.step}`;
  $("#seat-money").textContent = money.format(seat.money);
  $("#seat-shed").textContent = `${seat.shed_total}/100`;
  $("#seat-carried").textContent = money.format(seat.carried_total);
  $("#seat-failed").textContent = seat.action_summary.noop || 0;
  renderAlarms(seat.alarms);
  renderWorkers(seat.workers);
  renderMarket(seat.market);
  renderActionTicker(seat.workers);
}

function renderActionTicker(workers) {
  const ticker = $("#action-ticker");
  ticker.hidden = false;
  const active = workers.filter(worker => worker.op !== "PASS");
  const container = $("#action-chips");
  if (!active.length) {
    container.innerHTML = '<span class="action-chip">所有人员等待</span>';
    return;
  }
  container.innerHTML = active.map(worker => {
    const symbol = worker.status === "success" ? "✓" : worker.status === "noop" ? "!" : "?";
    return `<span class="action-chip ${escapeHtml(worker.status)}" title="${escapeHtml(worker.result)}">${escapeHtml(worker.label)} · ${escapeHtml(worker.action)} ${symbol}</span>`;
  }).join("");
}

function setDiagnosticsOpen(open) {
  $(".workspace").classList.toggle("diagnostics-open", open);
  $("#diagnostics").classList.toggle("collapsed", !open);
  $("#toggle-diagnostics").textContent = open ? "收起诊断" : "展开诊断";
  requestAnimationFrame(() => {
    applyViewerOverlays();
    setTimeout(applyViewerOverlays, 240);
  });
}

function renderAlarms(alarms) {
  const container = $("#alarms");
  const total = severityCount(alarms);
  $("#alarm-count").textContent = String(total);
  if (!alarms.length) {
    container.className = "card-list empty-list";
    container.textContent = "本回合没有报警";
    return;
  }
  container.className = "card-list";
  container.innerHTML = alarms.map(alarm => {
    const coords = (alarm.coords || []).slice(0, 12).map(c => `(${c.x},${c.y})`).join(" ");
    const more = (alarm.coords || []).length > 12 ? ` +${alarm.coords.length - 12}` : "";
    return `<article class="alarm-card ${escapeHtml(alarm.severity)}">
      <div class="card-head"><strong>${escapeHtml(alarm.title)}</strong><span>× ${alarm.count}</span></div>
      <div class="card-detail">${escapeHtml(alarm.detail)}</div>
      ${coords ? `<div class="coords">地块：${coords}${more}</div>` : ""}
    </article>`;
  }).join("");
}

function statusIcon(status) {
  const symbols = { success: "✓", noop: "!", uncertain: "?", pass: "·" };
  return `<i class="status-icon ${status}">${symbols[status] || "?"}</i>`;
}

function renderWorkers(workers) {
  const showPass = $("#show-pass").checked;
  const rows = workers.filter(worker => showPass || worker.op !== "PASS");
  const container = $("#workers");
  if (!rows.length) {
    container.className = "worker-list empty-list";
    container.textContent = showPass ? "没有人员动作" : "本回合所有人员均等待（勾选“显示等待”可查看）";
    return;
  }
  container.className = "worker-list";
  container.innerHTML = rows.map(worker => `<article class="worker-row ${escapeHtml(worker.status)}">
    <div class="worker-name"><strong>${escapeHtml(worker.label)}</strong><span>(${worker.start[0]},${worker.start[1]})${worker.end ? ` → (${worker.end[0]},${worker.end[1]})` : ""}</span></div>
    <div class="worker-action"><strong>${escapeHtml(worker.action)}</strong><span>目标：${escapeHtml(worker.target)}</span></div>
    <div class="worker-result">${statusIcon(worker.status)}${escapeHtml(worker.result)}</div>
  </article>`).join("");
}

function renderMarket(orders) {
  const container = $("#market-orders");
  $("#market-count").textContent = String(orders.length);
  if (!orders.length) {
    container.className = "card-list empty-list";
    container.textContent = "本回合没有市场指令";
    return;
  }
  container.className = "card-list";
  container.innerHTML = orders.map(order => `<article class="market-card ${escapeHtml(order.status)}">
    <div class="card-head"><strong>${order.index}. ${escapeHtml(order.action)}</strong><span>${statusIcon(order.status)}</span></div>
    <div class="card-detail">${escapeHtml(order.result)}</div>
  </article>`).join("");
}

function installOverlayStyle(doc) {
  if (doc.getElementById("diagnostic-overlay-style")) return;
  const style = doc.createElement("style");
  style.id = "diagnostic-overlay-style";
  style.textContent = `
    html, body, #app, .player, .viewer, .viewer > div,
    .viewer > div > div:has(> .game-renderer-isolation),
    .game-renderer-isolation, .kaggriculture-container {
      width:100% !important; height:100% !important; min-height:0 !important;
    }
    .viewer > div, .viewer > div > div:has(> .game-renderer-isolation) {
      flex:1 1 auto !important;
    }
    .kaggriculture-container {
      position:relative !important; overflow:hidden !important; padding:0 !important;
      background:#080c12 !important; background-image:none !important;
    }
    .kaggriculture-main {
      --diagnostic-fit-scale:1;
      position:absolute !important; left:50% !important; top:50% !important;
      width:calc(100% - 24px) !important; margin:0 !important;
      background-color:#aacf8a !important; background-repeat:repeat !important;
      background-size:80px 80px !important;
      transform:translate(-50%,-50%) scale(var(--diagnostic-fit-scale)) !important;
      transform-origin:center center !important;
    }
    .cell { position: relative !important; }
    .diag-action-bubble {
      --bubble-index: 0;
      position:absolute; z-index:95; left:50%; top:calc(-17px - var(--bubble-index) * 19px);
      transform:translateX(-50%); min-width:max-content; max-width:108px; height:18px;
      padding:3px 6px; border-radius:8px; color:#092217; background:#bff4d8;
      border:1px solid #35bb7c; font:800 9px/11px "Microsoft YaHei",sans-serif;
      box-shadow:0 2px 7px rgba(0,0,0,.56); white-space:nowrap; pointer-events:none;
    }
    .diag-action-bubble::after {
      content:""; position:absolute; left:50%; bottom:-4px; margin-left:-4px;
      border-left:4px solid transparent; border-right:4px solid transparent;
      border-top:4px solid #35bb7c;
    }
    .diag-action-bubble.noop { color:#3b080b; background:#ffc4c7; border-color:#e54952; }
    .diag-action-bubble.noop::after { border-top-color:#e54952; }
    .diag-action-bubble.uncertain { color:#392304; background:#ffe3a7; border-color:#d99a2c; }
    .diag-action-bubble.uncertain::after { border-top-color:#d99a2c; }
    .diag-entity-bubble {
      position:absolute; z-index:88; left:50%; bottom:3px; transform:translateX(-50%);
      min-width:max-content; padding:2px 5px; border-radius:7px; pointer-events:none;
      color:#3a2302; background:#ffe59e; border:1px solid #e0a72f;
      font:900 9px/11px "Microsoft YaHei",sans-serif;
      box-shadow:0 2px 7px rgba(0,0,0,.55); white-space:nowrap;
    }
    .diag-entity-bubble::after {
      content:""; position:absolute; left:50%; top:-4px; margin-left:-4px;
      border-left:4px solid transparent; border-right:4px solid transparent;
      border-bottom:4px solid #e0a72f;
    }
    .diag-entity-bubble.critical {
      color:white; background:#d93643; border-color:#ff737c;
      animation:diag-pulse .75s ease-in-out infinite alternate;
    }
    .diag-entity-bubble.critical::after { border-bottom-color:#d93643; }
    @keyframes diag-pulse { from { transform:translateX(-50%) scale(1); } to { transform:translateX(-50%) scale(1.08); } }
    .diag-alarm-critical { outline:3px solid #ff4f59 !important; outline-offset:-3px; }
    .diag-alarm-warning { box-shadow:inset 0 0 0 3px #f0ad43 !important; }
    .diag-alarm-info { box-shadow:inset 0 0 0 2px #42a5ff !important; }
  `;
  doc.head.appendChild(style);
}

function fitOfficialViewer(doc) {
  const container = doc.querySelector(".kaggriculture-container");
  const main = doc.querySelector(".kaggriculture-main");
  if (!container || !main) return;
  let ancestor = container;
  while (ancestor && ancestor !== doc.body) {
    ancestor.style.setProperty("width", "100%", "important");
    ancestor.style.setProperty("height", "100%", "important");
    ancestor.style.setProperty("min-height", "0", "important");
    ancestor.style.setProperty("max-height", "none", "important");
    ancestor.style.setProperty("flex", "1 1 auto", "important");
    ancestor = ancestor.parentElement;
  }
  const availableWidth = Math.max(1, container.clientWidth - 8);
  const availableHeight = Math.max(1, container.clientHeight - 8);
  const contentWidth = Math.max(1, main.offsetWidth);
  const contentHeight = Math.max(1, main.offsetHeight);
  const scale = Math.min(availableWidth / contentWidth, availableHeight / contentHeight);
  if (container.style.backgroundImage && container.style.backgroundImage !== "none") {
    main.style.backgroundImage = container.style.backgroundImage;
  }
  main.style.setProperty("--diagnostic-fit-scale", String(scale));
  main.dataset.diagnosticFitScale = scale.toFixed(4);
  viewer.dataset.fitScale = scale.toFixed(4);
  viewer.dataset.fitSize = `${contentWidth}x${contentHeight} into ${availableWidth}x${availableHeight}`;
}

function ensureViewerFitObserver(doc) {
  const win = doc.defaultView;
  const container = doc.querySelector(".kaggriculture-container");
  if (!win || !container || win.__diagnosticFitObserver) return;
  let pending = 0;
  const scheduleFit = () => {
    win.cancelAnimationFrame(pending);
    pending = win.requestAnimationFrame(() => fitOfficialViewer(doc));
  };
  const resizeObserver = new win.ResizeObserver(scheduleFit);
  resizeObserver.observe(container);
  const mutationObserver = new win.MutationObserver(scheduleFit);
  mutationObserver.observe(container, { childList: true, subtree: true });
  win.__diagnosticFitObserver = { resizeObserver, mutationObserver };
  scheduleFit();
  win.setTimeout(scheduleFit, 120);
  win.setTimeout(scheduleFit, 420);
  win.setTimeout(scheduleFit, 900);
}

function shortWorkerAction(worker) {
  const actor = worker.id === "farmer" ? "主" : `雇${String(worker.id).replace("hand_", "")}`;
  const command = worker.command || [];
  const op = worker.op;
  const labels = {
    NORTH: "移动↑", SOUTH: "移动↓", WEST: "移动←", EAST: "移动→",
    WATER: "浇水", HARVEST: "收获", FERTILIZE: "施肥", DIG: "挖除",
    BUILD_COOP: "建鸡舍", BUILD_PASTURE: "建牧场", FEED: "喂养",
    CARE: "照料", COLLECT_FERTILIZER: "收肥", PICKUP: "取货",
    DROP: "入仓", PLACE: command[1] ? `放${command[1]}` : "放置",
    PLANT: command[1] ? `种${command[1]}` : "种植",
  };
  const status = worker.status === "success" ? "✓" : worker.status === "noop" ? "✕" : "?";
  return `${actor} ${labels[op] || worker.action} ${status}`;
}

function applyViewerOverlays() {
  if (!state.frame || !state.viewerReady) return;
  let doc;
  try { doc = viewer.contentDocument; } catch (_) { return; }
  if (!doc) return;
  installOverlayStyle(doc);
  ensureViewerFitObserver(doc);
  fitOfficialViewer(doc);
  doc.querySelectorAll(".diag-action-bubble,.diag-entity-bubble").forEach(node => node.remove());
  doc.querySelectorAll(".diag-alarm-critical,.diag-alarm-warning,.diag-alarm-info").forEach(node => {
    node.classList.remove("diag-alarm-critical", "diag-alarm-warning", "diag-alarm-info");
  });
  state.frame.seats.forEach((seat, seatIndex) => {
    const bubbleCounts = new Map();
    seat.workers.filter(worker => worker.badge && worker.op !== "PASS").forEach(worker => {
      const position = worker.end || worker.start;
      const x = Number(position[0]);
      const y = Number(position[1]);
      const cell = doc.querySelector(`.farm-panel[data-player="${seatIndex + 1}"] .cell[data-row="${y}"][data-col="${x}"]`);
      if (!cell) return;
      const key = `${x},${y}`;
      const stackIndex = bubbleCounts.get(key) || 0;
      bubbleCounts.set(key, stackIndex + 1);
      const bubble = doc.createElement("span");
      bubble.className = `diag-action-bubble ${worker.status}`;
      bubble.style.setProperty("--bubble-index", String(stackIndex));
      bubble.textContent = shortWorkerAction(worker);
      bubble.title = `${worker.label}: ${worker.action} — ${worker.result}`;
      cell.appendChild(bubble);
    });
    const entityBubbleKeys = new Set(["plant_at_risk", "plant_critical", "animal_at_risk", "animal_critical"]);
    const entitySeen = new Set();
    seat.alarms.forEach(alarm => {
      (alarm.coords || []).forEach(({x, y}) => {
        const cell = doc.querySelector(`.farm-panel[data-player="${seatIndex + 1}"] .cell[data-row="${y}"][data-col="${x}"]`);
        if (!cell) return;
        cell.classList.add(`diag-alarm-${alarm.severity}`);
        const entityKey = `${x},${y}`;
        if (!entityBubbleKeys.has(alarm.key) || entitySeen.has(entityKey)) return;
        entitySeen.add(entityKey);
        const bubble = doc.createElement("span");
        bubble.className = `diag-entity-bubble ${alarm.severity}`;
        bubble.textContent = alarm.key.startsWith("plant_") ? "⚠ 快枯死" : "⚠ 快逃跑";
        bubble.title = alarm.detail;
        cell.appendChild(bubble);
      });
    });
  });
  doc.defaultView?.requestAnimationFrame(() => fitOfficialViewer(doc));
}

function playbackDelay() {
  return Math.max(40, 520 / Number($("#speed").value || 1));
}

function schedulePlayback() {
  clearTimeout(state.timer);
  if (!state.playing) return;
  state.timer = setTimeout(async () => {
    if (!state.meta || state.step >= state.meta.steps - 1) {
      stopPlayback();
      return;
    }
    await setStep(state.step + 1);
    schedulePlayback();
  }, playbackDelay());
}

function startPlayback() {
  if (!state.meta) return;
  if (state.step >= state.meta.steps - 1) setStep(0);
  state.playing = true;
  $("#play-pause").textContent = "Ⅱ";
  schedulePlayback();
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
  $("#play-pause").textContent = "▶";
}

function jumpNextCritical() {
  if (!state.timeline.length) return;
  const start = state.step + 1;
  const found = state.timeline.slice(start).find(row => row.seats[state.seat].severity === "critical");
  if (!found) return toast("后续没有严重报警");
  stopPlayback();
  setStep(found.step);
}

window.addEventListener("message", event => {
  if (event.source !== viewer.contentWindow || !event.data || typeof event.data !== "object") return;
  if (event.data.ready) {
    state.viewerReady = true;
    viewer.contentWindow.postMessage({ step: state.step, playing: false, parentHandlesUi: true, dense: true }, "*");
    setTimeout(applyViewerOverlays, 180);
  }
  if (typeof event.data.step === "number" && event.data.step !== state.step) {
    stopPlayback();
    setStep(event.data.step);
  }
});

viewer.addEventListener("load", () => {
  setTimeout(() => {
    state.viewerReady = true;
    if (viewer.contentWindow) viewer.contentWindow.postMessage({ step: state.step, playing: false, parentHandlesUi: true, dense: true }, "*");
    applyViewerOverlays();
  }, 250);
  [520, 900, 1400].forEach(delay => setTimeout(applyViewerOverlays, delay));
});

const viewerResizeObserver = new ResizeObserver(() => {
  requestAnimationFrame(applyViewerOverlays);
});
viewerResizeObserver.observe(viewer);
window.addEventListener("resize", () => requestAnimationFrame(applyViewerOverlays));

$("#load-path").addEventListener("click", loadPath);
$("#replay-path").addEventListener("keydown", event => { if (event.key === "Enter") loadPath(); });
$("#upload-file").addEventListener("change", event => uploadFile(event.target.files[0]));
$("#prev-step").addEventListener("click", () => { stopPlayback(); setStep(state.step - 1); });
$("#next-step").addEventListener("click", () => { stopPlayback(); setStep(state.step + 1); });
$("#play-pause").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
slider.addEventListener("input", event => { stopPlayback(); setStep(event.target.value); });
$("#speed").addEventListener("change", () => { if (state.playing) schedulePlayback(); });
$("#next-critical").addEventListener("click", jumpNextCritical);
$("#toggle-diagnostics").addEventListener("click", () => setDiagnosticsOpen($("#diagnostics").classList.contains("collapsed")));
$("#close-diagnostics").addEventListener("click", () => setDiagnosticsOpen(false));
$("#show-pass").addEventListener("change", () => state.frame && renderWorkers(state.frame.seats[state.seat].workers));
$$(".seat-tab").forEach(button => button.addEventListener("click", () => {
  state.seat = Number(button.dataset.seat);
  $$(".seat-tab").forEach(node => node.classList.toggle("active", node === button));
  if (state.frame) renderFrame();
}));

window.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); state.playing ? stopPlayback() : startPlayback(); }
  if (event.key === "ArrowLeft") { stopPlayback(); setStep(state.step - 1); }
  if (event.key === "ArrowRight") { stopPlayback(); setStep(state.step + 1); }
});

(async function bootstrap() {
  try {
    const data = await api("/api/meta");
    await activateReplay(data.meta);
    $("#replay-path").value = data.meta.path || "";
  } catch (_) {
    // Starting without --replay is valid; the user can load one from the UI.
  }
})();
