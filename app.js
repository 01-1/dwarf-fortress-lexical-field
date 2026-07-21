(() => {
  "use strict";

  const data = window.EMBEDDING_DATA;
  if (!data?.points?.length) throw new Error("Embedding data did not load.");

  const canvas = document.querySelector("#map");
  const ctx = canvas.getContext("2d", { alpha: true });
  const stage = document.querySelector(".stage");
  const points = data.points;
  const defaultIndex = Math.max(0, points.findIndex((point) => point.w === "cat" && point.p === "n"));

  const CLUSTER_COLORS = [
    "#70dfc4", "#e99b65", "#a8d46f", "#c88be0",
    "#f0c967", "#71a9ee", "#e7768e", "#8cd5e2",
    "#dc8264", "#9b9ee9", "#67cf8b", "#dbaa75",
    "#6fc0b7", "#bb82bb", "#d6d078", "#7899d9",
  ];
  const POS_COLORS = { n: "#62d9c0", v: "#f2b667", adj: "#c790e8", "?": "#7aa9f5" };
  const POS_NAMES = { n: "Noun", v: "Verb", adj: "Adjective", "?": "Other" };

  const state = {
    width: 0, height: 0, dpr: 1,
    zoom: 1, panX: 0, panY: 0,
    targetZoom: 1, targetPanX: 0, targetPanY: 0,
    pointerX: -1000, pointerY: -1000,
    hovered: defaultIndex, selected: null,
    dragging: false, dragMoved: false, dragStartX: 0, dragStartY: 0,
    panStartX: 0, panStartY: 0,
    activePos: new Set(["n", "v", "adj", "?"]),
    colorMode: "cluster", labels: true, motion: !matchMedia("(prefers-reduced-motion: reduce)").matches,
    start: performance.now(), needsDraw: true,
  };

  const byPos = points.reduce((acc, p) => ((acc[p.p] ||= 0), acc[p.p]++, acc), {});
  document.querySelector("#count-n").textContent = byPos.n || 0;
  document.querySelector("#count-v").textContent = byPos.v || 0;
  document.querySelector("#count-adj").textContent = byPos.adj || 0;
  document.querySelector("#count-other").textContent = byPos["?"] || 0;

  function colorFor(point) {
    return state.colorMode === "cluster" ? CLUSTER_COLORS[point.c % CLUSTER_COLORS.length] : POS_COLORS[point.p];
  }

  function baseScale() {
    return Math.max(175, Math.min(state.width, state.height) * .39);
  }

  function toScreen(point) {
    const scale = baseScale() * state.zoom;
    return {
      x: state.width * .5 + state.panX + point.x * scale,
      y: state.height * .5 + state.panY + point.y * scale,
    };
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    state.dpr = Math.min(devicePixelRatio || 1, 2);
    state.width = rect.width;
    state.height = rect.height;
    canvas.width = Math.round(rect.width * state.dpr);
    canvas.height = Math.round(rect.height * state.dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    state.needsDraw = true;
  }

  function hexToRgba(hex, alpha) {
    const value = parseInt(hex.slice(1), 16);
    return `rgba(${value >> 16},${(value >> 8) & 255},${value & 255},${alpha})`;
  }

  function roundRectPath(x, y, width, height, radius) {
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, radius);
  }

  function drawBackground() {
    ctx.clearRect(0, 0, state.width, state.height);

    const glow = ctx.createRadialGradient(state.width * .5, state.height * .48, 0, state.width * .5, state.height * .48, Math.max(state.width, state.height) * .64);
    glow.addColorStop(0, "rgba(31, 49, 58, .24)");
    glow.addColorStop(.55, "rgba(13, 22, 30, .08)");
    glow.addColorStop(1, "rgba(3, 6, 10, .3)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, state.width, state.height);

    const spacing = 74 * Math.max(.75, Math.min(state.zoom, 1.8));
    const offsetX = ((state.panX % spacing) + spacing) % spacing;
    const offsetY = ((state.panY % spacing) + spacing) % spacing;
    ctx.beginPath();
    for (let x = offsetX; x < state.width; x += spacing) { ctx.moveTo(x, 0); ctx.lineTo(x, state.height); }
    for (let y = offsetY; y < state.height; y += spacing) { ctx.moveTo(0, y); ctx.lineTo(state.width, y); }
    ctx.strokeStyle = "rgba(255,255,255,.018)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawConnections(focus) {
    if (focus == null || !state.activePos.has(points[focus].p)) return;
    const from = toScreen(points[focus]);
    points[focus].nn.slice(0, 8).forEach((neighborIndex, rank) => {
      const neighbor = points[neighborIndex];
      if (!state.activePos.has(neighbor.p)) return;
      const to = toScreen(neighbor);
      const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
      gradient.addColorStop(0, hexToRgba(colorFor(points[focus]), .44 - rank * .025));
      gradient.addColorStop(1, hexToRgba(colorFor(neighbor), .08));
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = gradient; ctx.lineWidth = rank < 3 ? 1.1 : .65; ctx.stroke();
    });
  }

  function drawClusterLabels() {
    if (!state.labels || state.zoom < .66) return;
    const centroids = data.clusters.map(() => ({ x: 0, y: 0, n: 0 }));
    points.forEach((point) => {
      if (!state.activePos.has(point.p)) return;
      centroids[point.c].x += point.x;
      centroids[point.c].y += point.y;
      centroids[point.c].n++;
    });
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    centroids.forEach((centroid, clusterId) => {
      if (!centroid.n) return;
      const screen = toScreen({ x: centroid.x / centroid.n, y: centroid.y / centroid.n });
      if (screen.x < -100 || screen.x > state.width + 100 || screen.y < -30 || screen.y > state.height + 30) return;
      const text = data.clusters[clusterId].name.toUpperCase();
      ctx.font = "700 8px Inter, system-ui, sans-serif";
      const width = ctx.measureText(text).width + 16;
      roundRectPath(screen.x - width / 2, screen.y - 11, width, 22, 5);
      ctx.fillStyle = "rgba(8,12,18,.68)"; ctx.fill();
      ctx.strokeStyle = hexToRgba(CLUSTER_COLORS[clusterId], .18); ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = hexToRgba(CLUSTER_COLORS[clusterId], .55);
      ctx.fillText(text, screen.x, screen.y + .5);
    });
  }

  function drawPoints(time) {
    const focus = state.selected ?? state.hovered;
    const neighborSet = focus == null ? null : new Set(points[focus].nn);
    const animatedTime = state.motion ? (time - state.start) / 1000 : 0;

    points.forEach((point, index) => {
      if (!state.activePos.has(point.p)) return;
      const screen = toScreen(point);
      if (screen.x < -15 || screen.x > state.width + 15 || screen.y < -15 || screen.y > state.height + 15) return;

      const isFocus = index === focus;
      const isNeighbor = neighborSet?.has(index);
      const dimmed = focus != null && !isFocus && !isNeighbor;
      const pulse = state.motion ? Math.sin(animatedTime * .9 + index * 1.713) * .22 : 0;
      const radius = isFocus ? 5.2 : isNeighbor ? 3.4 : Math.max(1.2, Math.min(2.5, 1.35 + state.zoom * .28 + pulse));
      const color = colorFor(point);

      if (isFocus) {
        const halo = ctx.createRadialGradient(screen.x, screen.y, 2, screen.x, screen.y, 20 + pulse * 4);
        halo.addColorStop(0, hexToRgba(color, .32)); halo.addColorStop(1, hexToRgba(color, 0));
        ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(screen.x, screen.y, 22, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(screen.x, screen.y, 10 + Math.sin(animatedTime * 2) * 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = hexToRgba(color, .32); ctx.lineWidth = 1; ctx.stroke();
      }

      ctx.beginPath(); ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, dimmed ? .1 : isFocus ? 1 : isNeighbor ? .86 : .56 + pulse * .16);
      ctx.fill();
      if (isFocus || isNeighbor) {
        ctx.strokeStyle = "rgba(245,247,243,.72)"; ctx.lineWidth = isFocus ? 1.3 : .6; ctx.stroke();
      }
    });

    if (focus != null) drawPointLabel(focus);
  }

  function drawPointLabel(index) {
    const point = points[index];
    const screen = toScreen(point);
    const text = point.w;
    ctx.font = "500 13px Georgia, serif";
    const width = ctx.measureText(text).width + 16;
    let x = screen.x + 12;
    if (x + width > state.width - 8) x = screen.x - width - 12;
    let y = screen.y - 13;
    if (y < 8) y = screen.y + 10;
    roundRectPath(x, y, width, 25, 5);
    ctx.fillStyle = "rgba(8,12,18,.94)"; ctx.fill();
    ctx.strokeStyle = hexToRgba(colorFor(point), .42); ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = "#edf0eb"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, x + width / 2, y + 12.5);
  }

  function render(time) {
    state.zoom += (state.targetZoom - state.zoom) * .16;
    state.panX += (state.targetPanX - state.panX) * .16;
    state.panY += (state.targetPanY - state.panY) * .16;
    const stillAnimating = Math.abs(state.targetZoom - state.zoom) > .001 || Math.abs(state.targetPanX - state.panX) > .1 || Math.abs(state.targetPanY - state.panY) > .1;
    if (!stage.hidden && (state.motion || state.needsDraw || stillAnimating)) {
      drawBackground();
      const focus = state.selected ?? state.hovered;
      drawConnections(focus);
      drawPoints(time);
      drawClusterLabels();
      state.needsDraw = false;
    }
    requestAnimationFrame(render);
  }

  function pointNear(x, y) {
    let closest = null;
    let closestDistance = Infinity;
    for (let index = 0; index < points.length; index++) {
      const point = points[index];
      if (!state.activePos.has(point.p)) continue;
      const screen = toScreen(point);
      if (screen.x < 0 || screen.x > state.width || screen.y < 0 || screen.y > state.height) continue;
      const distance = (screen.x - x) ** 2 + (screen.y - y) ** 2;
      if (distance < closestDistance) { closestDistance = distance; closest = index; }
    }
    return closest;
  }

  function updateInspector(index, pinned = false) {
    const empty = document.querySelector(".inspector-empty");
    const content = document.querySelector(".inspector-content");
    if (index == null) {
      content.hidden = true; empty.hidden = false; return;
    }
    const point = points[index];
    empty.hidden = true; content.hidden = false;
    document.querySelector("#detail-word").textContent = point.w;
    document.querySelector("#detail-pos").textContent = `${POS_NAMES[point.p]} · cluster ${String(point.c + 1).padStart(2, "0")}`;
    document.querySelector("#detail-dot").style.background = colorFor(point);
    document.querySelector("#detail-dot").style.color = colorFor(point);
    const repetitions = point.count > 1 ? ` · ${point.count} source entries` : "";
    document.querySelector("#detail-meta").textContent = `${data.clusters[point.c].name}${repetitions}${pinned ? " · pinned" : ""}`;
    const neighbors = document.querySelector("#neighbors");
    neighbors.replaceChildren(...point.nn.slice(0, 8).map((neighborIndex) => {
      const button = document.createElement("button");
      button.className = "neighbor"; button.type = "button";
      button.textContent = points[neighborIndex].w;
      button.addEventListener("click", () => selectPoint(neighborIndex, true));
      return button;
    }));
    document.querySelector("#estimated-note").hidden = !point.estimated;
  }

  function selectPoint(index, center = false) {
    state.selected = index;
    state.hovered = index;
    if (!state.activePos.has(points[index].p)) {
      state.activePos.add(points[index].p);
      syncFilters();
    }
    updateInspector(index, true);
    if (center) centerPoint(index);
    state.needsDraw = true;
  }

  function centerPoint(index) {
    const point = points[index];
    state.targetZoom = Math.max(state.targetZoom, 1.65);
    const scale = baseScale() * state.targetZoom;
    state.targetPanX = -point.x * scale;
    state.targetPanY = -point.y * scale;
    updateZoomReadout();
  }

  function updateZoomReadout() {
    document.querySelector("#zoom-readout").textContent = `${Math.round(state.targetZoom * 100)}%`;
  }

  function setZoom(next, anchorX = state.width / 2, anchorY = state.height / 2) {
    next = Math.max(.45, Math.min(16, next));
    const old = state.targetZoom;
    const ratio = next / old;
    state.targetPanX = anchorX - state.width / 2 - (anchorX - state.width / 2 - state.targetPanX) * ratio;
    state.targetPanY = anchorY - state.height / 2 - (anchorY - state.height / 2 - state.targetPanY) * ratio;
    state.targetZoom = next;
    updateZoomReadout(); state.needsDraw = true;
  }

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    state.dragging = true; state.dragMoved = false;
    state.dragStartX = event.clientX; state.dragStartY = event.clientY;
    state.panStartX = state.targetPanX; state.panStartY = state.targetPanY;
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    state.pointerX = event.clientX - rect.left; state.pointerY = event.clientY - rect.top;
    if (state.dragging) {
      const dx = event.clientX - state.dragStartX, dy = event.clientY - state.dragStartY;
      if (Math.abs(dx) + Math.abs(dy) > 3) state.dragMoved = true;
      state.targetPanX = state.panStartX + dx; state.targetPanY = state.panStartY + dy;
      state.needsDraw = true; return;
    }
    const nextHover = pointNear(state.pointerX, state.pointerY);
    if (nextHover !== state.hovered) {
      state.hovered = nextHover;
      if (state.selected == null) updateInspector(nextHover);
      state.needsDraw = true;
    }
  });
  function endPointer(event) {
    if (!state.dragging) return;
    if (!state.dragMoved) {
      const rect = canvas.getBoundingClientRect();
      const hit = pointNear(event.clientX - rect.left, event.clientY - rect.top);
      if (hit != null) selectPoint(hit);
      else { state.selected = null; updateInspector(state.hovered); }
    }
    state.dragging = false; canvas.classList.remove("dragging"); state.needsDraw = true;
  }
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", () => {
    if (!state.dragging) { state.hovered = null; if (state.selected == null) updateInspector(null); state.needsDraw = true; }
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    setZoom(state.targetZoom * Math.exp(-event.deltaY * .0012), event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });
  canvas.addEventListener("dblclick", (event) => {
    const rect = canvas.getBoundingClientRect();
    setZoom(state.targetZoom * 1.8, event.clientX - rect.left, event.clientY - rect.top);
  });

  function syncFilters() {
    document.querySelectorAll(".filter").forEach((button) => button.classList.toggle("active", state.activePos.has(button.dataset.pos)));
    const visible = points.filter((point) => state.activePos.has(point.p)).length;
    document.querySelector("#visible-count").textContent = `${visible.toLocaleString()} visible`;
    if (state.selected != null && !state.activePos.has(points[state.selected].p)) state.selected = null;
    state.needsDraw = true;
  }

  document.querySelectorAll(".filter").forEach((button) => button.addEventListener("click", () => {
    const pos = button.dataset.pos;
    state.activePos.has(pos) ? state.activePos.delete(pos) : state.activePos.add(pos);
    syncFilters();
  }));
  document.querySelectorAll(".segmented button").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".segmented button").forEach((item) => item.classList.toggle("active", item === button));
    state.colorMode = button.dataset.mode;
    if (state.selected != null) updateInspector(state.selected, true);
    state.needsDraw = true;
  }));

  document.querySelector("#labels-button").addEventListener("click", (event) => {
    state.labels = !state.labels;
    event.currentTarget.classList.toggle("active", state.labels);
    event.currentTarget.setAttribute("aria-pressed", state.labels);
    state.needsDraw = true;
  });
  document.querySelector("#motion-button").addEventListener("click", (event) => {
    state.motion = !state.motion;
    event.currentTarget.classList.toggle("active", state.motion);
    event.currentTarget.setAttribute("aria-pressed", state.motion);
    event.currentTarget.setAttribute("aria-label", state.motion ? "Pause motion" : "Resume motion");
    state.needsDraw = true;
  });
  document.querySelector("#reset-button").addEventListener("click", resetView);
  document.querySelector("#zoom-in").addEventListener("click", () => setZoom(state.targetZoom * 1.28));
  document.querySelector("#zoom-out").addEventListener("click", () => setZoom(state.targetZoom / 1.28));
  document.querySelector("#close-inspector").addEventListener("click", () => {
    state.selected = null; state.hovered = null; updateInspector(null); state.needsDraw = true;
  });

  function resetView() {
    state.targetZoom = 1; state.targetPanX = 0; state.targetPanY = 0;
    state.selected = null; state.hovered = defaultIndex;
    updateInspector(defaultIndex); updateZoomReadout(); state.needsDraw = true;
  }

  const search = document.querySelector("#search");
  const searchResults = document.querySelector("#search-results");
  let resultIndices = [];

  function runSearch() {
    const query = search.value.trim().toLowerCase();
    if (!query) { searchResults.classList.remove("open"); searchResults.replaceChildren(); return; }
    resultIndices = points
      .map((point, index) => ({ index, point, score: point.w === query ? 0 : point.w.startsWith(query) ? 1 : point.w.includes(query) ? 2 : 3 }))
      .filter((item) => item.score < 3)
      .sort((a, b) => a.score - b.score || a.point.w.length - b.point.w.length || a.point.w.localeCompare(b.point.w))
      .slice(0, 8)
      .map((item) => item.index);
    searchResults.replaceChildren(...resultIndices.map((index, resultPosition) => {
      const point = points[index];
      const button = document.createElement("button");
      button.className = `search-result${resultPosition === 0 ? " focused" : ""}`;
      button.type = "button"; button.setAttribute("role", "option");
      button.innerHTML = `<i></i><strong></strong><small></small>`;
      button.querySelector("i").style.background = colorFor(point);
      button.querySelector("strong").textContent = point.w;
      button.querySelector("small").textContent = POS_NAMES[point.p];
      button.addEventListener("click", () => chooseSearchResult(index));
      return button;
    }));
    searchResults.classList.toggle("open", resultIndices.length > 0);
  }

  function chooseSearchResult(index) {
    search.value = points[index].w;
    searchResults.classList.remove("open");
    if (!window.LexicalViews?.selectSearchResult(index)) selectPoint(index, true);
    search.blur();
  }

  search.addEventListener("input", runSearch);
  search.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && resultIndices.length) { event.preventDefault(); chooseSearchResult(resultIndices[0]); }
    if (event.key === "Escape") { search.value = ""; searchResults.classList.remove("open"); search.blur(); }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== search) { event.preventDefault(); search.focus(); }
    if (event.key === "Escape" && document.activeElement !== search && !window.LexicalViews?.handleEscape()) resetView();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".search-wrap")) searchResults.classList.remove("open");
  });

  new ResizeObserver(resize).observe(stage);
  resize();
  syncFilters();
  updateInspector(defaultIndex);
  window.LexicalMap = { resize, resetView, selectPoint };
  requestAnimationFrame(render);
})();
