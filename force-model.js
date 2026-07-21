(function (root, factory) {
  const model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  else root.FORCE_MODEL = model;
}(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  const EMPHASIZED_NEIGHBORS = 10;
  const EMPHASIZED_SHARE = .75;
  const LAYOUT_WEIGHT_REFRESH = 10;

  function makeTargetLookup(similarityBytes) {
    const histogram = new Uint32Array(256);
    similarityBytes.forEach((value) => histogram[value]++);
    const lookup = new Float64Array(256);
    let pairsAbove = 0;
    for (let value = 255; value >= 0; value--) {
      const averageDescendingRank = pairsAbove + (histogram[value] + 1) / 2;
      lookup[value] = Math.sqrt(averageDescendingRank / similarityBytes.length);
      pairsAbove += histogram[value];
    }
    return lookup;
  }

  function initializeForces(nodes) {
    nodes.forEach((node) => {
      node.fx = -node.x * .0014;
      node.fy = -node.y * .0014;
    });
  }

  function makeExponentialRankProfile(nodeCount) {
    const relationshipCount = Math.max(0, nodeCount - 1);
    if (!relationshipCount) return { weights: new Float64Array(0), decayBase: 0, topShare: 1 };
    const emphasized = Math.min(EMPHASIZED_NEIGHBORS, relationshipCount);
    let decayBase = 1;
    if (emphasized < relationshipCount && emphasized / relationshipCount < EMPHASIZED_SHARE) {
      let low = 0, high = 1;
      for (let iteration = 0; iteration < 64; iteration++) {
        const candidate = (low + high) / 2;
        const top = (1 - candidate ** emphasized) / (1 - candidate);
        const all = (1 - candidate ** relationshipCount) / (1 - candidate);
        if (top / all > EMPHASIZED_SHARE) low = candidate;
        else high = candidate;
      }
      decayBase = (low + high) / 2;
    }
    const weights = Float64Array.from(
      { length: relationshipCount },
      (_unused, rank) => decayBase ** rank,
    );
    const total = weights.reduce((sum, value) => sum + value, 0);
    const componentBudget = relationshipCount / 2;
    weights.forEach((value, rank) => { weights[rank] = value * componentBudget / total; });
    const emphasizedWeight = weights.slice(0, emphasized).reduce((sum, value) => sum + value, 0);
    return { weights, decayBase, topShare: emphasizedWeight / componentBudget };
  }

  function buildRankPairWeights(nodes, rankProfile, metricForPair) {
    const count = nodes.length;
    const directed = new Float32Array(count * count);
    for (let a = 0; a < count; a++) {
      const ranked = [];
      for (let b = 0; b < count; b++) {
        if (a !== b) ranked.push({ b, value: metricForPair(nodes[a], nodes[b]) });
      }
      ranked.sort((one, two) => one.value - two.value || one.b - two.b);
      for (let start = 0; start < ranked.length;) {
        let end = start + 1;
        while (end < ranked.length && ranked[end].value === ranked[start].value) end++;
        let tiedWeight = 0;
        for (let rank = start; rank < end; rank++) tiedWeight += rankProfile.weights[rank];
        tiedWeight /= end - start;
        for (let rank = start; rank < end; rank++) directed[a * count + ranked[rank].b] = tiedWeight;
        start = end;
      }
    }
    const pairs = new Float32Array(count * (count - 1) / 2);
    let offset = 0;
    for (let a = 0; a < count; a++) {
      for (let b = a + 1; b < count; b++) {
        pairs[offset++] = (directed[a * count + b] + directed[b * count + a]) / 2;
      }
    }
    return pairs;
  }

  function createWeightState(nodes, normalizedTargetForPair) {
    const rankProfile = makeExponentialRankProfile(nodes.length);
    return {
      rankProfile,
      semantic: buildRankPairWeights(
        nodes,
        rankProfile,
        (one, two) => normalizedTargetForPair(one.index, two.index),
      ),
      layout: buildRankPairWeights(
        nodes,
        rankProfile,
        (one, two) => Math.hypot(two.x - one.x, two.y - one.y),
      ),
      steps: 0,
    };
  }

  function refreshLayoutWeights(nodes, weightState) {
    if (weightState.steps > 0 && weightState.steps % LAYOUT_WEIGHT_REFRESH === 0) {
      weightState.layout = buildRankPairWeights(
        nodes,
        weightState.rankProfile,
        (one, two) => Math.hypot(two.x - one.x, two.y - one.y),
      );
    }
  }

  function heapPush(heap, entry, limit) {
    if (heap.length < limit) {
      heap.push(entry);
      let index = heap.length - 1;
      while (index > 0) {
        const parent = (index - 1) >> 1;
        if (heap[parent].distance >= heap[index].distance) break;
        [heap[parent], heap[index]] = [heap[index], heap[parent]];
        index = parent;
      }
      return;
    }
    if (entry.distance >= heap[0].distance) return;
    heap[0] = entry;
    let index = 0;
    while (true) {
      const left = index * 2 + 1, right = left + 1;
      if (left >= heap.length) break;
      const largest = right < heap.length && heap[right].distance > heap[left].distance ? right : left;
      if (heap[index].distance >= heap[largest].distance) break;
      [heap[index], heap[largest]] = [heap[largest], heap[index]];
      index = largest;
    }
  }

  function buildKdTree(nodes, indices = nodes.map((_node, index) => index), depth = 0) {
    if (!indices.length) return null;
    const axis = depth & 1;
    indices.sort((a, b) => (axis ? nodes[a].y - nodes[b].y : nodes[a].x - nodes[b].x) || a - b);
    const middle = indices.length >> 1;
    return {
      index: indices[middle], axis,
      left: buildKdTree(nodes, indices.slice(0, middle), depth + 1),
      right: buildKdTree(nodes, indices.slice(middle + 1), depth + 1),
    };
  }

  function nearestFromKdTree(nodes, tree, queryIndex, limit) {
    const query = nodes[queryIndex], heap = [];
    function visit(branch) {
      if (!branch) return;
      const candidate = nodes[branch.index];
      const axisDelta = branch.axis ? query.y - candidate.y : query.x - candidate.x;
      const near = axisDelta < 0 ? branch.left : branch.right;
      const far = axisDelta < 0 ? branch.right : branch.left;
      visit(near);
      if (branch.index !== queryIndex) {
        const dx = candidate.x - query.x, dy = candidate.y - query.y;
        heapPush(heap, { index: branch.index, distance: dx * dx + dy * dy }, limit);
      }
      if (heap.length < limit || axisDelta * axisDelta < heap[0].distance) visit(far);
    }
    visit(tree);
    return heap.sort((a, b) => a.distance - b.distance || a.index - b.index).map((entry) => entry.index);
  }

  function buildSparsePairs(nodes, semanticDirected, layoutDirected) {
    const selected = new Map();
    for (let a = 0; a < nodes.length; a++) {
      const candidates = new Set([
        ...semanticDirected[a].keys(),
        ...layoutDirected[a].keys(),
      ]);
      candidates.forEach((b) => {
        const weight = (
          (semanticDirected[a].get(b) || 0) + (semanticDirected[b].get(a) || 0)
          + (layoutDirected[a].get(b) || 0) + (layoutDirected[b].get(a) || 0)
        ) / 2;
        const left = Math.min(a, b), right = Math.max(a, b), key = `${left}:${right}`;
        const existing = selected.get(key);
        if (!existing || weight > existing.weight) selected.set(key, { a: left, b: right, weight });
      });
    }
    return [...selected.values()];
  }

  function createApproximateWeightState(nodes, normalizedTargetForPair, semanticCandidatesForNode) {
    const rankProfile = makeExponentialRankProfile(nodes.length);
    const forceNeighborCount = Math.max(1, Math.ceil(Math.sqrt(nodes.length)));
    const candidateCount = Math.min(nodes.length - 1, forceNeighborCount);
    const localByGlobal = new Map(nodes.map((node, index) => [node.index, index]));
    const semanticDirected = nodes.map((node) => {
      const result = new Map();
      for (const globalIndex of semanticCandidatesForNode(node)) {
        const localIndex = localByGlobal.get(globalIndex);
        if (localIndex == null || localIndex === localByGlobal.get(node.index) || result.has(localIndex)) continue;
        result.set(localIndex, rankProfile.weights[result.size]);
        if (result.size >= candidateCount) break;
      }
      return result;
    });
    const weightState = {
      approximate: true,
      rankProfile,
      forceNeighborCount,
      candidateCount,
      semanticDirected,
      layoutDirected: null,
      pairs: [],
      steps: 0,
    };
    refreshApproximateWeightState(nodes, weightState, true);
    return weightState;
  }

  function refreshApproximateWeightState(nodes, weightState, force = false) {
    if (!force && (weightState.steps === 0 || weightState.steps % LAYOUT_WEIGHT_REFRESH !== 0)) return;
    const tree = buildKdTree(nodes);
    weightState.layoutDirected = nodes.map((_node, index) => {
      const result = new Map();
      nearestFromKdTree(nodes, tree, index, weightState.candidateCount).forEach((neighbor, rank) => {
        result.set(neighbor, weightState.rankProfile.weights[rank]);
      });
      return result;
    });
    weightState.pairs = buildSparsePairs(
      nodes,
      weightState.semanticDirected,
      weightState.layoutDirected,
    );
  }

  function applyExactRepulsion(nodes, idealDistance, heat) {
    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        const one = nodes[a], two = nodes[b];
        let dx = two.x - one.x, dy = two.y - one.y;
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < .01) {
          dx = ((a * 17 + b * 31) % 7 - 3) * .1;
          dy = ((a * 29 + b * 13) % 7 - 3) * .1;
          distanceSquared = Math.max(.0001, dx * dx + dy * dy);
        }
        const force = Math.min(.04, (idealDistance * idealDistance / distanceSquared) * .018) * heat;
        const forceX = dx * force, forceY = dy * force;
        one.fx -= forceX; one.fy -= forceY;
        two.fx += forceX; two.fy += forceY;
      }
    }
  }

  function applySemanticPair(nodes, a, b, pairWeight, idealDistance, strength, heatFactor, normalizedTargetForPair) {
    const one = nodes[a], two = nodes[b];
    let dx = two.x - one.x, dy = two.y - one.y;
    const distance = Math.max(.01, Math.hypot(dx, dy));
    if (distance < .02) {
      dx = .01 + (a % 3) * .005;
      dy = .01 + (b % 3) * .005;
    }
    const targetDistance = idealDistance * normalizedTargetForPair(one.index, two.index);
    const force = pairWeight * (distance - targetDistance) * strength * heatFactor;
    const forceX = dx / distance * force, forceY = dy / distance * force;
    one.fx += forceX; one.fy += forceY;
    two.fx -= forceX; two.fy -= forceY;
  }

  function applySemanticStress(nodes, idealDistance, heat, normalizedTargetForPair, weightState) {
    const strength = .04 / Math.max(1, nodes.length), heatFactor = .45 + heat * .55;
    let pairOffset = 0;
    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        const pairWeight = weightState.semantic[pairOffset] + weightState.layout[pairOffset];
        pairOffset++;
        applySemanticPair(nodes, a, b, pairWeight, idealDistance, strength, heatFactor, normalizedTargetForPair);
      }
    }
  }

  function applySparseSemanticStress(nodes, idealDistance, heat, normalizedTargetForPair, weightState) {
    const strength = .04 / Math.max(1, nodes.length), heatFactor = .45 + heat * .55;
    weightState.pairs.forEach(({ a, b, weight }) => {
      applySemanticPair(nodes, a, b, weight, idealDistance, strength, heatFactor, normalizedTargetForPair);
    });
  }

  function integrate(nodes) {
    nodes.forEach((node) => {
      if (node.fixed) {
        node.vx = 0; node.vy = 0;
        return;
      }
      node.vx = (node.vx + node.fx) * .86;
      node.vy = (node.vy + node.fy) * .86;
      node.x += node.vx; node.y += node.vy;
    });
  }

  function stepExact(nodes, idealDistance, heat, normalizedTargetForPair, weightState) {
    refreshLayoutWeights(nodes, weightState);
    initializeForces(nodes);
    applyExactRepulsion(nodes, idealDistance, heat);
    applySemanticStress(nodes, idealDistance, heat, normalizedTargetForPair, weightState);
    integrate(nodes);
    weightState.steps++;
    return Math.max(.09, heat * .992);
  }

  return {
    EMPHASIZED_NEIGHBORS,
    EMPHASIZED_SHARE,
    LAYOUT_WEIGHT_REFRESH,
    makeTargetLookup,
    makeExponentialRankProfile,
    createWeightState,
    refreshLayoutWeights,
    createApproximateWeightState,
    refreshApproximateWeightState,
    initializeForces,
    applyExactRepulsion,
    applySemanticStress,
    applySparseSemanticStress,
    integrate,
    stepExact,
  };
}));
