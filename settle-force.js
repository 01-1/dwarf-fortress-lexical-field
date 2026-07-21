#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const ForceModel = require("./force-model.js");

const root = __dirname;
const embeddingPath = path.join(root, "embedding-data.js");
const similarityPath = path.join(root, "all-pair-similarity.js");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(embeddingPath, "utf8"), context);
vm.runInNewContext(fs.readFileSync(similarityPath, "utf8"), context);

const data = context.window.EMBEDDING_DATA;
const encoded = context.window.ALL_PAIR_SIMILARITY;
const similarityBytes = Uint8Array.from(Buffer.from(encoded.data, "base64"));
const targetLookup = ForceModel.makeTargetLookup(similarityBytes);
const count = data.points.length;
const pairOffset = (one, two) => {
  const a = Math.min(one, two), b = Math.max(one, two);
  return a * (2 * count - a - 1) / 2 + (b - a - 1);
};
const normalizedTargetForPair = (one, two) => targetLookup[similarityBytes[pairOffset(one, two)]];
const idealDistance = Math.max(12, 82 * Math.sqrt(42 / Math.max(42, count)));

const nodes = data.points.map((point, index) => {
  const angle = index * 2.399963229728653;
  return {
    index,
    x: point.x * idealDistance * .5 + Math.cos(angle) * idealDistance * 1e-4,
    y: point.y * idealDistance * .5 + Math.sin(angle) * idealDistance * 1e-4,
    vx: 0, vy: 0, fx: 0, fy: 0, fixed: false,
  };
});
const meanX = nodes.reduce((sum, node) => sum + node.x, 0) / count;
const meanY = nodes.reduce((sum, node) => sum + node.y, 0) / count;
nodes.forEach((node) => { node.x -= meanX; node.y -= meanY; });

let heat = 1;
let iterations = 0;
let settledSteps = 0;
const requiredSettledSteps = 40;
const velocityTolerance = idealDistance * 2e-3;
const p99VelocityTolerance = idealDistance * 8e-3;
let rmsVelocity = Infinity, p99Velocity = Infinity, maximumVelocity = Infinity;
const weightState = ForceModel.createWeightState(nodes, normalizedTargetForPair);

while (settledSteps < requiredSettledSteps) {
  heat = ForceModel.stepExact(nodes, idealDistance, heat, normalizedTargetForPair, weightState);
  iterations++;
  const speeds = nodes.map((node) => Math.hypot(node.vx, node.vy)).sort((a, b) => a - b);
  rmsVelocity = Math.sqrt(speeds.reduce((sum, speed) => sum + speed * speed, 0) / count);
  p99Velocity = speeds[Math.min(count - 1, Math.floor((count - 1) * .99))];
  maximumVelocity = speeds[count - 1];
  settledSteps = rmsVelocity < velocityTolerance && p99Velocity < p99VelocityTolerance
    ? settledSteps + 1
    : 0;
  if (!Number.isFinite(rmsVelocity)) throw new Error("Full force layout diverged before settling");
  if (iterations % 100 === 0) {
    console.log(`Force layout iteration ${iterations}: RMS velocity ${rmsVelocity.toFixed(6)}, p99 ${p99Velocity.toFixed(6)}, max ${maximumVelocity.toFixed(6)}`);
  }
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

const medianX = median(nodes.map((node) => node.x));
const medianY = median(nodes.map((node) => node.y));
nodes.forEach((node) => { node.x -= medianX; node.y -= medianY; });
const radii = nodes.map((node) => Math.hypot(node.x, node.y)).sort((a, b) => a - b);
const normalizationRadius = radii[Math.floor((radii.length - 1) * .99)];
nodes.forEach((node, index) => {
  data.points[index].fx = Number((node.x / normalizationRadius).toFixed(5));
  data.points[index].fy = Number((node.y / normalizationRadius).toFixed(5));
});
data.meta.graphLayout = "settled exact all-pairs shared JavaScript force simulation";
data.meta.graphDiagnostics = {
  iterations,
  settledSteps,
  requiredSettledSteps,
  rmsVelocity: Number(rmsVelocity.toFixed(8)),
  p99Velocity: Number(p99Velocity.toFixed(8)),
  maximumVelocity: Number(maximumVelocity.toFixed(8)),
  velocityTolerance,
  p99VelocityTolerance,
  finalHeat: heat,
  repulsion: "exact O(n^2)",
  semanticForce: "exact O(n^2)",
  implementation: "force-model.js",
  allNodesMovable: true,
  pairWeight: "additive exponential ranks for semantic and current-layout closeness",
  emphasizedNeighbors: ForceModel.EMPHASIZED_NEIGHBORS,
  emphasizedWeightShare: ForceModel.EMPHASIZED_SHARE,
  actualEmphasizedWeightShare: Number(weightState.rankProfile.topShare.toFixed(8)),
  exponentialDecayBase: Number(weightState.rankProfile.decayBase.toFixed(12)),
  layoutWeightRefresh: ForceModel.LAYOUT_WEIGHT_REFRESH,
  normalizationRadius: Number(normalizationRadius.toFixed(8)),
};

fs.writeFileSync(embeddingPath, `window.EMBEDDING_DATA=${JSON.stringify(data)};\n`);
console.log(`Settled ${count} words in ${iterations} iterations with shared JavaScript forces`);
