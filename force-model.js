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

  function applySemanticStress(nodes, idealDistance, heat, normalizedTargetForPair, weightState) {
    const strength = .04 / Math.max(1, nodes.length);
    const heatFactor = .45 + heat * .55;
    let pairOffset = 0;
    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        const one = nodes[a], two = nodes[b];
        let dx = two.x - one.x, dy = two.y - one.y;
        const distance = Math.max(.01, Math.hypot(dx, dy));
        if (distance < .02) {
          dx = .01 + (a % 3) * .005;
          dy = .01 + (b % 3) * .005;
        }
        const normalizedTarget = normalizedTargetForPair(one.index, two.index);
        const targetDistance = idealDistance * normalizedTarget;
        const error = distance - targetDistance;
        const pairWeight = weightState.semantic[pairOffset] + weightState.layout[pairOffset];
        pairOffset++;
        const force = pairWeight * error * strength * heatFactor;
        const forceX = dx / distance * force, forceY = dy / distance * force;
        one.fx += forceX; one.fy += forceY;
        two.fx -= forceX; two.fy -= forceY;
      }
    }
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
    initializeForces,
    applyExactRepulsion,
    applySemanticStress,
    integrate,
    stepExact,
  };
}));
