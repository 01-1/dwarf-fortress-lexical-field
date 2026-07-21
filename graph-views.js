(() => {
  "use strict";

  const data = window.EMBEDDING_DATA;
  const points = data.points;
  const forceModel = window.FORCE_MODEL;
  if (!forceModel) throw new Error("Shared force model did not load.");
  const similarityPayload = window.ALL_PAIR_SIMILARITY;
  if (!similarityPayload || similarityPayload.count !== points.length) throw new Error("All-pair similarity data did not load.");
  const similarityBinary = atob(similarityPayload.data);
  const allPairSimilarity = Uint8Array.from(similarityBinary, (character) => character.charCodeAt(0));
  const inverseSquareRankDistance = forceModel.makeTargetLookup(allPairSimilarity);
  const COLORS = [
    "#70dfc4", "#e99b65", "#a8d46f", "#c88be0",
    "#f0c967", "#71a9ee", "#e7768e", "#8cd5e2",
    "#dc8264", "#9b9ee9", "#67cf8b", "#dbaa75",
    "#6fc0b7", "#bb82bb", "#d6d078", "#7899d9",
  ];
  const POS_NAMES = { n: "noun", v: "verb", adj: "adjective", "?": "other" };
  const graphEdges = [];
  const seenEdges = new Set();

  points.forEach((point, index) => point.nn.forEach((neighbor) => {
    const a = Math.min(index, neighbor), b = Math.max(index, neighbor);
    const key = `${a}:${b}`;
    if (!seenEdges.has(key)) { seenEdges.add(key); graphEdges.push([a, b]); }
  }));

  const initialIndex = Math.max(0, points.findIndex((point) => point.w === "cat" && point.p === "n"));
  let activeView = "map";

  function similarityByteForPair(one, two) {
    if (one === two) return 255;
    const a = Math.min(one, two), b = Math.max(one, two), count = points.length;
    const offset = a * (2 * count - a - 1) / 2 + (b - a - 1);
    return allPairSimilarity[offset];
  }

  function rgba(hex, alpha) {
    const value = parseInt(hex.slice(1), 16);
    return `rgba(${value >> 16},${(value >> 8) & 255},${value & 255},${alpha})`;
  }

  function fitCanvas(canvas, context) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: rect.width, height: rect.height, dpr };
  }

  function drawWordLabel(context, text, x, y, color, emphasized = false) {
    context.font = `${emphasized ? 500 : 400} ${emphasized ? 14 : 10}px Georgia, serif`;
    const width = context.measureText(text).width + (emphasized ? 16 : 10);
    const height = emphasized ? 25 : 18;
    context.beginPath();
    context.roundRect(x - width / 2, y - height / 2, width, height, emphasized ? 5 : 4);
    context.fillStyle = emphasized ? "rgba(8,12,18,.94)" : "rgba(8,12,18,.76)";
    context.fill();
    context.strokeStyle = rgba(color, emphasized ? .52 : .2);
    context.lineWidth = 1;
    context.stroke();
    context.fillStyle = emphasized ? "#f3f0e8" : "rgba(225,228,222,.72)";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, x, y + .5);
  }

  function updateFocus(prefix, index, choose) {
    const point = points[index];
    document.querySelector(`#${prefix}-word`).textContent = point.w;
    document.querySelector(`#${prefix}-meta`).textContent = `${POS_NAMES[point.p]} · ${data.clusters[point.c].name}`;
    const holder = document.querySelector(`#${prefix}-neighbors`);
    holder.replaceChildren(...point.nn.slice(0, 6).map((neighborIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = points[neighborIndex].w;
      button.addEventListener("click", () => choose(neighborIndex));
      return button;
    }));
  }

  // -------------------------------------------------------------------------
  // Settled force view: final state of the full live-force model run offline.
  // -------------------------------------------------------------------------
  const staticCanvas = document.querySelector("#static-map");
  const staticContext = staticCanvas.getContext("2d", { alpha: true });
  const similarityCanvas = document.querySelector("#similarity-chart");
  const similarityContext = similarityCanvas.getContext("2d", { alpha: true });
  const ascendingCanvas = document.querySelector("#ascending-chart");
  const ascendingContext = ascendingCanvas.getContext("2d", { alpha: true });
  const similarityOrder = data.similarityOrder;
  let similarityHover = null, ascendingHover = null;
  const staticState = {
    width: 0, height: 0, zoom: 1, panX: 0, panY: 0,
    selected: initialIndex, hovered: null, dragging: false, moved: false,
    dragX: 0, dragY: 0, startPanX: 0, startPanY: 0, frame: 0,
  };

  function staticScale() { return Math.max(180, Math.min(staticState.width, staticState.height) * .46) * staticState.zoom; }
  function staticScreen(point) {
    const scale = staticScale();
    return {
      x: staticState.width * .54 + staticState.panX + point.fx * scale,
      y: staticState.height * .5 + staticState.panY + point.fy * scale,
    };
  }

  function resizeStatic() {
    Object.assign(staticState, fitCanvas(staticCanvas, staticContext));
    scheduleStaticDraw();
  }

  function drawSimilarityCurves() {
    if (!similarityOrder?.squareLargest?.length || !similarityOrder?.ascending?.length) return;
    drawOrderCurve(similarityCanvas, similarityContext, similarityOrder.squareLargest, similarityHover, "#89e5c2");
    drawOrderCurve(ascendingCanvas, ascendingContext, similarityOrder.ascending, ascendingHover, "#f1b865");
  }

  function drawOrderCurve(canvas, context, samples, hover, color) {
    const size = fitCanvas(canvas, context);
    const width = size.width, height = size.height;
    if (!width || !height) return;
    const padding = { top: 7, right: 5, bottom: 9, left: 26 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    let minimum = -.05, maximum = .05;
    samples.forEach((value) => { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); });
    const xAt = (index) => padding.left + index / (samples.length - 1) * plotWidth;
    const yAt = (value) => padding.top + (maximum - value) / (maximum - minimum) * plotHeight;

    context.clearRect(0, 0, width, height);
    context.font = "7px ui-monospace, monospace";
    context.textAlign = "right";
    context.textBaseline = "middle";
    [minimum, 0, maximum].forEach((value) => {
      const y = yAt(value);
      context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y);
      context.strokeStyle = value === 0 ? "rgba(255,255,255,.11)" : "rgba(255,255,255,.045)";
      context.lineWidth = 1; context.stroke();
      context.fillStyle = "rgba(122,133,146,.7)";
      context.fillText(value.toFixed(2), padding.left - 4, y);
    });

    const fill = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    fill.addColorStop(0, rgba(color, .18)); fill.addColorStop(1, rgba(color, 0));
    context.beginPath();
    samples.forEach((value, index) => {
      const x = xAt(index), y = yAt(value);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.lineTo(width - padding.right, height - padding.bottom);
    context.lineTo(padding.left, height - padding.bottom);
    context.closePath(); context.fillStyle = fill; context.fill();

    context.beginPath();
    samples.forEach((value, index) => {
      const x = xAt(index), y = yAt(value);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.strokeStyle = rgba(color, .78);
    context.lineWidth = 1.25; context.stroke();

    if (hover != null) {
      const x = xAt(hover), y = yAt(samples[hover]);
      context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom);
      context.strokeStyle = "rgba(255,255,255,.22)"; context.lineWidth = 1; context.stroke();
      context.beginPath(); context.arc(x, y, 3.2, 0, Math.PI * 2);
      context.fillStyle = "#f3f0e8"; context.fill();
    }
  }

  similarityCanvas.addEventListener("pointermove", (event) => {
    const rect = similarityCanvas.getBoundingClientRect();
    const plotLeft = 26, plotWidth = rect.width - 31;
    similarityHover = Math.max(0, Math.min(
      similarityOrder.squareLargest.length - 1,
      Math.round(((event.clientX - rect.left - plotLeft) / plotWidth) * (similarityOrder.squareLargest.length - 1)),
    ));
    const k = similarityHover + 1, rank = k * k, value = similarityOrder.squareLargest[similarityHover];
    document.querySelector("#similarity-readout").textContent = `k ${k.toLocaleString()} · ${rank.toLocaleString()}th largest · ${value.toFixed(4)}`;
    drawSimilarityCurves();
  });
  similarityCanvas.addEventListener("pointerleave", () => {
    similarityHover = null;
    document.querySelector("#similarity-readout").textContent = `${similarityOrder.pairCount.toLocaleString()} pairs`;
    drawSimilarityCurves();
  });
  ascendingCanvas.addEventListener("pointermove", (event) => {
    const rect = ascendingCanvas.getBoundingClientRect();
    const plotLeft = 26, plotWidth = rect.width - 31;
    ascendingHover = Math.max(0, Math.min(
      similarityOrder.ascending.length - 1,
      Math.round(((event.clientX - rect.left - plotLeft) / plotWidth) * (similarityOrder.ascending.length - 1)),
    ));
    const rank = 1 + Math.round(ascendingHover / (similarityOrder.ascending.length - 1) * (similarityOrder.pairCount - 1));
    const value = similarityOrder.ascending[ascendingHover];
    document.querySelector("#ascending-readout").textContent = `rank ${rank.toLocaleString()} · ${value.toFixed(4)}`;
    drawSimilarityCurves();
  });
  ascendingCanvas.addEventListener("pointerleave", () => {
    ascendingHover = null;
    document.querySelector("#ascending-readout").textContent = "display only";
    drawSimilarityCurves();
  });

  function drawStatic() {
    staticState.frame = 0;
    const { width, height } = staticState;
    staticContext.clearRect(0, 0, width, height);
    const focus = staticState.hovered ?? staticState.selected;
    const focusPoint = points[focus];
    const neighbors = new Set(focusPoint.nn);

    staticContext.lineWidth = .55;
    staticContext.strokeStyle = "rgba(160,177,181,.026)";
    staticContext.beginPath();
    graphEdges.forEach(([a, b]) => {
      const from = staticScreen(points[a]), to = staticScreen(points[b]);
      if ((from.x < -10 && to.x < -10) || (from.x > width + 10 && to.x > width + 10) ||
          (from.y < -10 && to.y < -10) || (from.y > height + 10 && to.y > height + 10)) return;
      staticContext.moveTo(from.x, from.y); staticContext.lineTo(to.x, to.y);
    });
    staticContext.stroke();

    const focusScreen = staticScreen(focusPoint);
    focusPoint.nn.forEach((neighborIndex, rank) => {
      const to = staticScreen(points[neighborIndex]);
      staticContext.beginPath();
      staticContext.moveTo(focusScreen.x, focusScreen.y); staticContext.lineTo(to.x, to.y);
      staticContext.strokeStyle = rgba(COLORS[focusPoint.c], .45 - rank * .035);
      staticContext.lineWidth = rank < 3 ? 1.15 : .7;
      staticContext.stroke();
    });

    points.forEach((point, index) => {
      const screen = staticScreen(point);
      if (screen.x < -8 || screen.x > width + 8 || screen.y < -8 || screen.y > height + 8) return;
      const isFocus = index === focus, isNeighbor = neighbors.has(index);
      const radius = isFocus ? 5 : isNeighbor ? 3 : Math.min(2.2, 1.1 + staticState.zoom * .22);
      staticContext.beginPath(); staticContext.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      staticContext.fillStyle = rgba(COLORS[point.c], isFocus ? 1 : isNeighbor ? .82 : .34);
      staticContext.fill();
    });

    focusPoint.nn.slice(0, staticState.zoom > 2 ? 8 : 5).forEach((index) => {
      const screen = staticScreen(points[index]);
      drawWordLabel(staticContext, points[index].w, screen.x, screen.y - 13, COLORS[points[index].c]);
    });
    drawWordLabel(staticContext, focusPoint.w, focusScreen.x, focusScreen.y - 18, COLORS[focusPoint.c], true);
  }

  function scheduleStaticDraw() {
    if (staticState.frame || activeView !== "static") return;
    staticState.frame = requestAnimationFrame(drawStatic);
  }

  function closestStatic(x, y) {
    let result = staticState.selected, best = Infinity;
    points.forEach((point, index) => {
      const screen = staticScreen(point);
      if (screen.x < 0 || screen.x > staticState.width || screen.y < 0 || screen.y > staticState.height) return;
      const distance = (screen.x - x) ** 2 + (screen.y - y) ** 2;
      if (distance < best) { best = distance; result = index; }
    });
    return result;
  }

  function selectStatic(index, center = false) {
    staticState.selected = index; staticState.hovered = null;
    updateFocus("static", index, (next) => selectStatic(next, true));
    if (center) {
      staticState.zoom = Math.max(1.7, staticState.zoom);
      const scale = staticScale();
      staticState.panX = -points[index].fx * scale;
      staticState.panY = -points[index].fy * scale;
    }
    scheduleStaticDraw();
  }

  staticCanvas.addEventListener("pointerdown", (event) => {
    staticCanvas.setPointerCapture(event.pointerId);
    staticState.dragging = true; staticState.moved = false;
    staticState.dragX = event.clientX; staticState.dragY = event.clientY;
    staticState.startPanX = staticState.panX; staticState.startPanY = staticState.panY;
    staticCanvas.classList.add("dragging");
  });
  staticCanvas.addEventListener("pointermove", (event) => {
    const rect = staticCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    if (staticState.dragging) {
      const dx = event.clientX - staticState.dragX, dy = event.clientY - staticState.dragY;
      if (Math.abs(dx) + Math.abs(dy) > 3) staticState.moved = true;
      staticState.panX = staticState.startPanX + dx; staticState.panY = staticState.startPanY + dy;
    } else {
      const closest = closestStatic(x, y);
      if (closest !== staticState.hovered) {
        staticState.hovered = closest;
        updateFocus("static", closest, (next) => selectStatic(next, true));
      }
    }
    scheduleStaticDraw();
  });
  function endStaticPointer(event) {
    if (!staticState.dragging) return;
    if (!staticState.moved) {
      const rect = staticCanvas.getBoundingClientRect();
      selectStatic(closestStatic(event.clientX - rect.left, event.clientY - rect.top));
    }
    staticState.dragging = false; staticCanvas.classList.remove("dragging"); scheduleStaticDraw();
  }
  staticCanvas.addEventListener("pointerup", endStaticPointer);
  staticCanvas.addEventListener("pointercancel", endStaticPointer);
  staticCanvas.addEventListener("pointerleave", () => {
    if (!staticState.dragging) { staticState.hovered = null; updateFocus("static", staticState.selected, (next) => selectStatic(next, true)); scheduleStaticDraw(); }
  });
  staticCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = staticCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const oldZoom = staticState.zoom;
    const nextZoom = Math.max(.45, Math.min(16, oldZoom * Math.exp(-event.deltaY * .0012)));
    const ratio = nextZoom / oldZoom;
    staticState.panX = x - staticState.width * .54 - (x - staticState.width * .54 - staticState.panX) * ratio;
    staticState.panY = y - staticState.height * .5 - (y - staticState.height * .5 - staticState.panY) * ratio;
    staticState.zoom = nextZoom; scheduleStaticDraw();
  }, { passive: false });
  document.querySelector("#static-zoom-in").addEventListener("click", () => { staticState.zoom = Math.min(16, staticState.zoom * 1.3); scheduleStaticDraw(); });
  document.querySelector("#static-zoom-out").addEventListener("click", () => { staticState.zoom = Math.max(.45, staticState.zoom / 1.3); scheduleStaticDraw(); });
  document.querySelector("#static-reset").addEventListener("click", resetStatic);
  function resetStatic() { staticState.zoom = 1; staticState.panX = 0; staticState.panY = 0; selectStatic(initialIndex); }

  // -------------------------------------------------------------------------
  // Live view: the same spring/repulsion model, continuously integrated in JS.
  // -------------------------------------------------------------------------
  const physicsCanvas = document.querySelector("#physics-map");
  const physicsContext = physicsCanvas.getContext("2d", { alpha: true });
  const physicsStage = document.querySelector("#physics-view");
  const physicsState = {
    width: 0, height: 0, anchor: initialIndex, nodes: [], edges: [], byIndex: new Map(),
    hovered: null, dragging: null, dragMoved: false, pointerX: 0, pointerY: 0,
    paused: false, heat: 1, frame: 0, lastTime: 0, nodeLimit: 42, sizeFrame: 0,
    zoom: 1, panX: 0, panY: 0, panning: false, panStartX: 0, panStartY: 0,
    repulsionMode: "auto", activeRepulsion: "exact", weightState: null,
  };

  function physicsCenter() {
    return { x: physicsState.width * .55 + physicsState.panX, y: physicsState.height * .5 + physicsState.panY };
  }

  function buildPhysics(anchor, resetViewport = false) {
    physicsState.anchor = anchor;
    const included = [anchor];
    const includedSet = new Set(included);
    const depthByIndex = new Map([[anchor, 0]]);
    const queue = [anchor];
    let queueCursor = 0;
    while (queueCursor < queue.length && included.length < physicsState.nodeLimit) {
      const parent = queue[queueCursor++];
      const nextDepth = depthByIndex.get(parent) + 1;
      for (const index of points[parent].nn) {
        if (includedSet.has(index)) continue;
        includedSet.add(index); included.push(index); queue.push(index);
        depthByIndex.set(index, nextDepth);
        if (included.length >= physicsState.nodeLimit) break;
      }
    }
    if (included.length < physicsState.nodeLimit) {
      const anchorPoint = points[anchor];
      const remaining = points
        .map((point, index) => ({ index, distance: (point.fx - anchorPoint.fx) ** 2 + (point.fy - anchorPoint.fy) ** 2 }))
        .filter(({ index }) => !includedSet.has(index))
        .sort((a, b) => a.distance - b.distance);
      for (const { index } of remaining) {
        includedSet.add(index); included.push(index); depthByIndex.set(index, 3);
        if (included.length >= physicsState.nodeLimit) break;
      }
    }
    const firstRing = new Set(points[anchor].nn);

    const anchorPoint = points[anchor];
    const layoutScale = Math.max(105, 520 * Math.sqrt(42 / Math.max(42, physicsState.nodeLimit)));
    physicsState.nodes = included.map((index, order) => {
      if (index === anchor) return { index, x: 0, y: 0, vx: 0, vy: 0, fx: 0, fy: 0, fixed: false, ring: 0 };
      const point = points[index];
      let x = (point.fx - anchorPoint.fx) * layoutScale;
      let y = (point.fy - anchorPoint.fy) * layoutScale;
      if (Math.hypot(x, y) < 35) {
        const angle = order * 2.399963;
        const radius = firstRing.has(index) ? 105 : 175;
        x = Math.cos(angle) * radius; y = Math.sin(angle) * radius;
      }
      return { index, x, y, vx: 0, vy: 0, fx: 0, fy: 0, fixed: false, ring: depthByIndex.get(index) };
    });
    physicsState.byIndex = new Map(physicsState.nodes.map((node, index) => [node.index, index]));
    physicsState.weightState = forceModel.createWeightState(
      physicsState.nodes,
      (one, two) => inverseSquareRankDistance[similarityByteForPair(one, two)],
    );
    const edgeKeys = new Set();
    physicsState.edges = [];
    physicsState.nodes.forEach((node, localIndex) => points[node.index].nn.forEach((neighborIndex, neighborRank) => {
      const other = physicsState.byIndex.get(neighborIndex);
      if (other == null) return;
      const a = Math.min(localIndex, other), b = Math.max(localIndex, other), key = `${a}:${b}`;
      if (a === b || edgeKeys.has(key)) return;
      edgeKeys.add(key);
      const direct = node.index === anchor || neighborIndex === anchor;
      const similarity = points[node.index].ns?.[neighborRank] ?? .5;
      physicsState.edges.push({ a, b, weight: Math.max(similarity, .05), direct });
    }));
    physicsState.heat = 1; physicsState.hovered = null;
    if (resetViewport) { physicsState.zoom = 1; physicsState.panX = 0; physicsState.panY = 0; }
    updateFocus("physics", anchor, (next) => buildPhysics(next, true));
  }

  function stepPhysics() {
    const nodes = physicsState.nodes;
    const idealDistance = Math.max(12, 82 * Math.sqrt(42 / Math.max(42, nodes.length)));
    forceModel.refreshLayoutWeights(nodes, physicsState.weightState);
    forceModel.initializeForces(nodes);

    // Exact and Barnes–Hut both approximate the same Fruchterman–Reingold
    // repulsive force. This selector affects computation, not semantic edges.
    const method = resolveRepulsionMethod(nodes.length);
    if (method === "exact") forceModel.applyExactRepulsion(nodes, idealDistance, physicsState.heat);
    else applyBarnesHutRepulsion(nodes, idealDistance);
    forceModel.applySemanticStress(
      nodes,
      idealDistance,
      physicsState.heat,
      (one, two) => inverseSquareRankDistance[similarityByteForPair(one, two)],
      physicsState.weightState,
    );
    forceModel.integrate(nodes);
    physicsState.weightState.steps++;
    physicsState.heat = Math.max(.09, physicsState.heat * .992);
  }

  function resolveRepulsionMethod(nodeCount) {
    const resolved = physicsState.repulsionMode === "auto"
      ? (nodeCount <= 300 ? "exact" : "barnes-hut")
      : physicsState.repulsionMode;
    if (resolved !== physicsState.activeRepulsion) {
      physicsState.activeRepulsion = resolved;
      updateRepulsionStatus();
    }
    return resolved;
  }

  function updateRepulsionStatus() {
    const label = physicsState.activeRepulsion === "exact" ? "exact O(n²)" : "Barnes–Hut O(n log n)";
    const automatic = physicsState.repulsionMode === "auto" ? "Automatic · " : "";
    document.querySelector("#physics-method-status").textContent = `${automatic}${label} repulsion · forces O(n²) · ranks O(n² log n) / 10 steps`;
  }

  function repelPair(one, two, a, b, idealDistance, symmetric) {
    let dx = two.x - one.x, dy = two.y - one.y;
    let distanceSquared = dx * dx + dy * dy;
    if (distanceSquared < .01) {
      dx = ((a * 17 + b * 31) % 7 - 3) * .1;
      dy = ((a * 29 + b * 13) % 7 - 3) * .1;
      distanceSquared = Math.max(.0001, dx * dx + dy * dy);
    }
    const force = Math.min(.04, (idealDistance * idealDistance / distanceSquared) * .018) * physicsState.heat;
    const forceX = dx * force, forceY = dy * force;
    one.fx -= forceX; one.fy -= forceY;
    if (symmetric) { two.fx += forceX; two.fy += forceY; }
  }

  function applyBarnesHutRepulsion(nodes, idealDistance) {
    const tree = buildQuadTree(nodes);
    nodes.forEach((node, index) => repelFromCell(node, index, tree, idealDistance));
  }

  function buildQuadTree(nodes) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodes.forEach((node) => {
      minX = Math.min(minX, node.x); minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x); maxY = Math.max(maxY, node.y);
    });
    const size = Math.max(maxX - minX, maxY - minY, 1);
    const root = makeCell((minX + maxX) / 2, (minY + maxY) / 2, size / 2 + 1);
    nodes.forEach((_node, index) => insertIntoTree(root, nodes, index, 0));
    return root;
  }

  function makeCell(x, y, half) {
    return { x, y, half, mass: 0, sumX: 0, sumY: 0, bodies: [], children: null };
  }

  function insertIntoTree(cell, nodes, index, depth) {
    const node = nodes[index];
    cell.mass++; cell.sumX += node.x; cell.sumY += node.y;
    if (!cell.children && (cell.bodies.length === 0 || depth >= 18)) {
      cell.bodies.push(index); return;
    }
    if (!cell.children) {
      const quarter = cell.half / 2;
      cell.children = [
        makeCell(cell.x - quarter, cell.y - quarter, quarter),
        makeCell(cell.x + quarter, cell.y - quarter, quarter),
        makeCell(cell.x - quarter, cell.y + quarter, quarter),
        makeCell(cell.x + quarter, cell.y + quarter, quarter),
      ];
      const previous = cell.bodies; cell.bodies = [];
      previous.forEach((body) => insertIntoChild(cell, nodes, body, depth));
    }
    insertIntoChild(cell, nodes, index, depth);
  }

  function insertIntoChild(cell, nodes, index, depth) {
    const node = nodes[index];
    const quadrant = (node.x >= cell.x ? 1 : 0) + (node.y >= cell.y ? 2 : 0);
    insertIntoTree(cell.children[quadrant], nodes, index, depth + 1);
  }

  function repelFromCell(node, index, cell, idealDistance) {
    if (!cell.mass) return;
    if (!cell.children) {
      cell.bodies.forEach((otherIndex) => {
        if (otherIndex !== index) repelPair(node, physicsState.nodes[otherIndex], index, otherIndex, idealDistance, false);
      });
      return;
    }

    const centerX = cell.sumX / cell.mass, centerY = cell.sumY / cell.mass;
    const dx = centerX - node.x, dy = centerY - node.y;
    const distanceSquared = Math.max(.01, dx * dx + dy * dy);
    const containsNode = Math.abs(node.x - cell.x) <= cell.half && Math.abs(node.y - cell.y) <= cell.half;
    const openingRatio = (cell.half * 2) / Math.sqrt(distanceSquared);
    if (!containsNode && openingRatio < .72) {
      const force = Math.min(.32, Math.min(.04, (idealDistance * idealDistance / distanceSquared) * .018) * cell.mass) * physicsState.heat;
      node.fx -= dx * force; node.fy -= dy * force;
      return;
    }
    cell.children.forEach((child) => repelFromCell(node, index, child, idealDistance));
  }

  function drawPhysics() {
    const { width, height, nodes } = physicsState;
    const center = physicsCenter();
    physicsContext.clearRect(0, 0, width, height);
    const focusNode = physicsState.hovered == null ? nodes[0] : nodes[physicsState.hovered];

    physicsState.edges.forEach((edge) => {
      const one = nodes[edge.a], two = nodes[edge.b];
      const connected = edge.a === physicsState.hovered || edge.b === physicsState.hovered || (physicsState.hovered == null && edge.direct);
      physicsContext.beginPath();
      physicsContext.moveTo(center.x + one.x * physicsState.zoom, center.y + one.y * physicsState.zoom);
      physicsContext.lineTo(center.x + two.x * physicsState.zoom, center.y + two.y * physicsState.zoom);
      physicsContext.strokeStyle = connected ? "rgba(137,229,194,.34)" : "rgba(174,188,191,.08)";
      physicsContext.lineWidth = connected ? 1.15 : .65;
      physicsContext.stroke();
    });

    nodes.forEach((node, localIndex) => {
      const point = points[node.index];
      const x = center.x + node.x * physicsState.zoom, y = center.y + node.y * physicsState.zoom;
      const focused = node === focusNode, direct = node.ring === 1;
      physicsContext.beginPath(); physicsContext.arc(x, y, focused ? 6 : direct ? 3.4 : 2.2, 0, Math.PI * 2);
      physicsContext.fillStyle = rgba(COLORS[point.c], focused ? 1 : direct ? .78 : .46);
      physicsContext.fill();
      if (focused) {
        physicsContext.beginPath(); physicsContext.arc(x, y, 12 + physicsState.heat * 3, 0, Math.PI * 2);
        physicsContext.strokeStyle = rgba(COLORS[point.c], .34); physicsContext.lineWidth = 1; physicsContext.stroke();
      }
      if (focused || direct || (physicsState.dragging === localIndex)) {
        drawWordLabel(physicsContext, point.w, x, y - (focused ? 20 : 14), COLORS[point.c], focused);
      }
    });
  }

  function physicsLoop(time) {
    physicsState.frame = requestAnimationFrame(physicsLoop);
    if (activeView !== "physics") return;
    if (!physicsState.paused && time - physicsState.lastTime > 12) { stepPhysics(); physicsState.lastTime = time; }
    drawPhysics();
  }

  function resizePhysics() { Object.assign(physicsState, fitCanvas(physicsCanvas, physicsContext)); }
  function closestPhysics(x, y, maximum = Infinity) {
    const center = physicsCenter();
    let result = null, best = maximum * maximum;
    physicsState.nodes.forEach((node, index) => {
      const distance = (center.x + node.x * physicsState.zoom - x) ** 2 + (center.y + node.y * physicsState.zoom - y) ** 2;
      if (distance < best) { best = distance; result = index; }
    });
    return result;
  }

  physicsCanvas.addEventListener("pointerdown", (event) => {
    const rect = physicsCanvas.getBoundingClientRect();
    physicsState.pointerX = event.clientX - rect.left; physicsState.pointerY = event.clientY - rect.top;
    if (event.shiftKey) {
      physicsCanvas.setPointerCapture(event.pointerId);
      physicsState.panning = true;
      physicsState.panStartX = physicsState.panX; physicsState.panStartY = physicsState.panY;
      physicsCanvas.classList.add("dragging");
      return;
    }
    physicsState.dragging = closestPhysics(physicsState.pointerX, physicsState.pointerY, 40);
    if (physicsState.dragging == null) return;
    physicsCanvas.setPointerCapture(event.pointerId);
    physicsState.dragMoved = false;
    physicsState.nodes[physicsState.dragging].fixed = true;
    physicsState.heat = Math.max(.55, physicsState.heat);
    physicsCanvas.classList.add("dragging");
  });
  physicsCanvas.addEventListener("pointermove", (event) => {
    const rect = physicsCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    if (physicsState.panning) {
      physicsState.panX = physicsState.panStartX + x - physicsState.pointerX;
      physicsState.panY = physicsState.panStartY + y - physicsState.pointerY;
    } else if (physicsState.dragging != null) {
      if (Math.hypot(x - physicsState.pointerX, y - physicsState.pointerY) > 3) physicsState.dragMoved = true;
      const center = physicsCenter(), node = physicsState.nodes[physicsState.dragging];
      node.x = (x - center.x) / physicsState.zoom; node.y = (y - center.y) / physicsState.zoom; node.vx = 0; node.vy = 0;
      physicsState.heat = Math.max(.45, physicsState.heat);
    } else {
      const closest = closestPhysics(x, y);
      if (closest !== physicsState.hovered) {
        physicsState.hovered = closest;
        updateFocus("physics", physicsState.nodes[closest].index, (next) => buildPhysics(next));
      }
    }
  });
  function endPhysicsPointer() {
    if (physicsState.panning) {
      physicsState.panning = false; physicsCanvas.classList.remove("dragging"); return;
    }
    if (physicsState.dragging == null) return;
    const dragged = physicsState.dragging;
    const node = physicsState.nodes[dragged];
    if (!physicsState.dragMoved) buildPhysics(node.index, true);
    else node.fixed = false;
    physicsState.dragging = null; physicsCanvas.classList.remove("dragging");
  }
  physicsCanvas.addEventListener("pointerup", endPhysicsPointer);
  physicsCanvas.addEventListener("pointercancel", endPhysicsPointer);
  physicsCanvas.addEventListener("pointerleave", () => {
    if (physicsState.dragging == null && !physicsState.panning) { physicsState.hovered = null; updateFocus("physics", physicsState.anchor, (next) => buildPhysics(next, true)); }
  });
  physicsCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = physicsCanvas.getBoundingClientRect();
    setPhysicsZoom(
      physicsState.zoom * Math.exp(-event.deltaY * .0012),
      event.clientX - rect.left,
      event.clientY - rect.top,
    );
  }, { passive: false });

  function setPhysicsZoom(nextZoom, anchorX = physicsState.width * .55, anchorY = physicsState.height * .5) {
    nextZoom = Math.max(.2, Math.min(16, nextZoom));
    const baseX = physicsState.width * .55, baseY = physicsState.height * .5;
    const worldX = (anchorX - baseX - physicsState.panX) / physicsState.zoom;
    const worldY = (anchorY - baseY - physicsState.panY) / physicsState.zoom;
    physicsState.panX = anchorX - baseX - worldX * nextZoom;
    physicsState.panY = anchorY - baseY - worldY * nextZoom;
    physicsState.zoom = nextZoom;
  }

  document.querySelector("#physics-toggle").addEventListener("click", (event) => {
    physicsState.paused = !physicsState.paused;
    physicsStage.classList.toggle("paused", physicsState.paused);
    event.currentTarget.textContent = physicsState.paused ? "Resume" : "Pause";
  });
  document.querySelector("#physics-reheat").addEventListener("click", () => { physicsState.heat = 1; physicsState.paused = false; physicsStage.classList.remove("paused"); document.querySelector("#physics-toggle").textContent = "Pause"; });
  const physicsSize = document.querySelector("#physics-size");
  const physicsSizeValue = document.querySelector("#physics-size-value");
  physicsSize.addEventListener("input", () => {
    physicsState.nodeLimit = Number(physicsSize.value);
    physicsSizeValue.textContent = `${physicsState.nodeLimit.toLocaleString()} words`;
    resolveRepulsionMethod(physicsState.nodeLimit);
    cancelAnimationFrame(physicsState.sizeFrame);
    physicsState.sizeFrame = requestAnimationFrame(() => buildPhysics(physicsState.anchor));
  });
  const physicsMethod = document.querySelector("#physics-method");
  physicsMethod.addEventListener("change", () => {
    physicsState.repulsionMode = physicsMethod.value;
    physicsState.activeRepulsion = resolveRepulsionMethod(physicsState.nodes.length);
    physicsState.heat = 1;
    updateRepulsionStatus();
  });
  document.querySelector("#physics-zoom-in").addEventListener("click", () => setPhysicsZoom(physicsState.zoom * 1.3));
  document.querySelector("#physics-zoom-out").addEventListener("click", () => setPhysicsZoom(physicsState.zoom / 1.3));
  document.querySelector("#physics-fit").addEventListener("click", () => {
    physicsState.zoom = 1; physicsState.panX = 0; physicsState.panY = 0;
  });

  // -------------------------------------------------------------------------
  // Shared tabs and search routing.
  // -------------------------------------------------------------------------
  const footerStatus = document.querySelector("footer strong");
  const footerMethod = document.querySelector("#footer-method");
  const headerActions = document.querySelector(".header-actions");
  const viewNames = { static: "Settled full-field force layout", physics: "Live physics simulation", similarity: "All-pairs similarity curve", map: "Semantic map ready" };
  const viewMethods = {
    static: "Exact all-pairs forces → settled snapshot",
    physics: "JavaScript forces → live all-pairs stress",
    similarity: "2,390,391 cosine values → square ranks",
    map: "GloVe 6B / 50D → t-SNE / cosine",
  };

  function activateView(view) {
    activeView = view;
    document.querySelectorAll(".view-tab").forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("active", active); tab.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".view-panel").forEach((panel) => { panel.hidden = panel.id !== `${view}-view`; });
    headerActions.classList.toggle("context-hidden", view !== "map");
    footerStatus.textContent = viewNames[view];
    footerMethod.textContent = viewMethods[view];
    if (view === "static") { resizeStatic(); scheduleStaticDraw(); }
    if (view === "physics") { resizePhysics(); physicsState.heat = Math.max(.45, physicsState.heat); }
    if (view === "similarity") drawSimilarityCurves();
    if (view === "map") window.LexicalMap?.resize();
  }

  document.querySelectorAll(".view-tab").forEach((tab) => tab.addEventListener("click", () => activateView(tab.dataset.view)));
  document.querySelector(".view-tabs").addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    const tabs = [...document.querySelectorAll(".view-tab")];
    const current = tabs.findIndex((tab) => tab.dataset.view === activeView);
    const offset = event.key === 'ArrowRight' ? 1 : -1;
    const next = tabs[(current + offset + tabs.length) % tabs.length];
    next.focus(); activateView(next.dataset.view);
  });

  window.LexicalViews = {
    selectSearchResult(index) {
      if (activeView === "static") { selectStatic(index, true); return true; }
      if (activeView === "physics") { buildPhysics(index, true); return true; }
      if (activeView === "similarity") { activateView("map"); window.LexicalMap?.selectPoint(index, true); return true; }
      return false;
    },
    handleEscape() {
      if (activeView === "static") { resetStatic(); return true; }
      if (activeView === "physics") { buildPhysics(initialIndex, true); return true; }
      if (activeView === "similarity") { similarityHover = null; ascendingHover = null; drawSimilarityCurves(); return true; }
      return false;
    },
  };

  new ResizeObserver(() => {
    if (activeView === "static") resizeStatic();
    if (activeView === "physics") resizePhysics();
    if (activeView === "similarity") drawSimilarityCurves();
  }).observe(document.querySelector(".workspace"));

  selectStatic(initialIndex);
  buildPhysics(initialIndex);
  resizeStatic();
  resizePhysics();
  activateView("map");
  requestAnimationFrame(physicsLoop);
})();
