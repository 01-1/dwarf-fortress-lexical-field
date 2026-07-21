(function (root, factory) {
  const model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  else root.FORCE_MODEL = model;
}(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  const PROXIMITY_MIN = 0;
  const PROXIMITY_MAX = 100;

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

  function applySemanticStress(nodes, idealDistance, heat, normalizedTargetForPair) {
    const strength = .04 / Math.max(1, nodes.length);
    const heatFactor = .45 + heat * .55;
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
        const sammonWeight = 1 / (normalizedTarget + .001);
        const rawProximityWeight = idealDistance / distance;
        const proximityWeight = Math.min(PROXIMITY_MAX, Math.max(PROXIMITY_MIN, rawProximityWeight));
        const proximityDerivative = rawProximityWeight === proximityWeight
          ? -idealDistance / (distance * distance)
          : 0;
        const error = distance - targetDistance;
        const weightedError = (sammonWeight + proximityWeight) * error
          + .5 * proximityDerivative * error * error;
        const force = weightedError * strength * heatFactor;
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

  function stepExact(nodes, idealDistance, heat, normalizedTargetForPair) {
    initializeForces(nodes);
    applyExactRepulsion(nodes, idealDistance, heat);
    applySemanticStress(nodes, idealDistance, heat, normalizedTargetForPair);
    integrate(nodes);
    return Math.max(.09, heat * .992);
  }

  return {
    PROXIMITY_MIN,
    PROXIMITY_MAX,
    makeTargetLookup,
    initializeForces,
    applyExactRepulsion,
    applySemanticStress,
    integrate,
    stepExact,
  };
}));
