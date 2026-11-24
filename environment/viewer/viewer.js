// This page shows a real season in 3D, and the local server plays each week with the real environment.

// The plot has 12 columns and 8 rows of plants.
const NUM_COLUMNS = 12;
const NUM_ROWS = 8;
const PLANT_COUNT = NUM_COLUMNS * NUM_ROWS;
const WEEKS = 18;

// This list is the same as in the Python code and shows how much of the harvest each week can add.
const STAGE_WEIGHTS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.07, 0.08, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.0];

// Keep a number between a lowest and a highest value.
function clamp(value, lowest = 0, highest = 1) {
  return Math.max(lowest, Math.min(highest, value));
}

// Turn a number between 0 and 1 into a smooth curve.
function smooth(value) {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value * value * (3 - 2 * value);
}

// Go from one number to another, where 0 is the first number and 1 is the second.
function lerp(from, to, amount) {
  return from + (to - from) * amount;
}

function $(id) {
  return document.getElementById(id);
}

// Show a number with thousands separators, for example 60,000.
function formatNumber(value) {
  return Math.round(value).toLocaleString("en-US");
}

// This small random number generator always gives the same numbers for the same seed.
function makeRandom(seed) {
  let state = seed >>> 0;
  return function () {
    state |= 0;
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// These values describe the season that is played now, and steps has one entry for each finished week.
let steps = [];
let startState = null;
let seasonDone = false;
let comparison = null;
let failurePenalty = null;
let nextObs = [];
// The notes and the chances for the coming decision, and the kind of farmer that is playing.
let nextNotes = [];
let nextChances = null;
let chanceKind = "chance";
let trainedFarmer = false;
let config = null;
let seasonLabel = "";

// These values are the choices of the user, which start from the address of the page.
const urlParams = new URLSearchParams(location.search);
let mode = urlParams.get("mode") === "play" ? "play" : "watch";
let model = urlParams.get("model") || "ppo";
let seed = parseInt(urlParams.get("seed") || "0", 10);
let test = urlParams.get("test");

// These values describe the animation, where one frame is one thirtieth of a second.
let frame = 0;
let playing = false;
let autoRun = false;
let busy = false;
let stopFrame = -1;
let advice = -1;
let lastHudKey = "";
let errorText = "";

const FPS = 30;
const INTRO_SECONDS = 0.5;
const SECONDS_PER_WEEK = 3.6;
const OUTRO_SECONDS = 6;

// Send a request to the local server and return its answer.
async function callApi(path, body) {
  let options = {};
  if (body !== undefined) {
    options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error);
  }
  return data;
}

// Turn one week from the server into the names used in this page.
function readStep(raw) {
  return {
    week: raw.week,
    stage: raw.stage,
    action: raw.a,
    actionName: raw.action,
    cost: raw.cost,
    reward: raw.reward,
    totalReturn: raw.ret,
    rain: raw.rain,
    moistureBefore: raw.moist_before,
    moistureAfter: raw.moist_after,
    budgetBefore: raw.budget_before,
    budgetAfter: raw.budget_after,
    before: raw.before,
    after: raw.after,
    flagsAfterAction: raw.flags_after_action,
    removedNow: raw.removed_now,
    obs: raw.obs,
    end: raw.end,
    kg: raw.kg,
    notes: raw.notes || [],
    judge: raw.judge || null,
    good: raw.good || 0,
    judged: raw.judged || 0,
    chances: raw.chances || null,
    modelPick: raw.model_pick,
    weedsBefore: raw.weeds_before || 0,
    weedsAfter: raw.weeds_after || 0,
  };
}

// The first frame of a week, where week 0 is the first week.
function weekStartFrame(weekIndex) {
  return Math.round((INTRO_SECONDS + weekIndex * SECONDS_PER_WEEK) * FPS);
}

// This is the last frame we can play now, and it grows as new weeks arrive.
function lastFrameAvailable() {
  let outroSeconds = 0;
  if (seasonDone) {
    outroSeconds = OUTRO_SECONDS;
  }
  const frames = Math.round((INTRO_SECONDS + steps.length * SECONDS_PER_WEEK + outroSeconds) * FPS) - 1;
  return Math.max(0, frames);
}

// Say which week we are in at this frame, and the panels use this.
function whereAmI(frameNumber) {
  const time = frameNumber / FPS;
  const weekCount = steps.length;
  let weekIndex = -1;
  let phase = 0;
  let isOutro = false;

  if (time >= INTRO_SECONDS && weekCount > 0) {
    weekIndex = Math.floor((time - INTRO_SECONDS) / SECONDS_PER_WEEK);
    phase = ((time - INTRO_SECONDS) % SECONDS_PER_WEEK) / SECONDS_PER_WEEK;
    if (weekIndex >= weekCount) {
      weekIndex = weekCount - 1;
      phase = 1;
      isOutro = seasonDone;
    }
  }
  return { time, weekIndex, phase, isOutro };
}


const renderer = new THREE.WebGLRenderer({ canvas: $("c"), antialias: true, preserveDrawingBuffer: true });
renderer.setSize(1280, 720, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1280 / 720, 0.1, 250);

scene.add(new THREE.HemisphereLight(0xdfeeff, 0x4a5a2c, 0.5));

const sun = new THREE.DirectionalLight(0xfff1d6, 0.85);
sun.position.set(-8, 14, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 40 });
scene.add(sun);

scene.fog = new THREE.Fog(0x9fb7c6, 40, 95);

// Objects we reuse many times to place things in the scene.
const matrix = new THREE.Matrix4();
const hiddenMatrix = new THREE.Matrix4().makeScale(0, 0, 0);
const quaternion = new THREE.Quaternion();
const euler = new THREE.Euler();
const position = new THREE.Vector3();
const scaleVector = new THREE.Vector3();
const partColor = new THREE.Color();
const leafColor = new THREE.Color();
const matureColor = new THREE.Color();

// The ground is olive green because it is the rainy season.
const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x56693a, roughness: 1 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(220, 220), groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// The plot is a brown rectangle that gets darker when the soil is wet.
const plotMaterial = new THREE.MeshStandardMaterial({ color: 0x8a6440, roughness: 1 });
const plot = new THREE.Mesh(new THREE.BoxGeometry(13.4, 0.08, 9.4), plotMaterial);
plot.position.y = 0.04;
plot.receiveShadow = true;
scene.add(plot);

// One raised ridge of soil for each row.
const ridgeMaterial = new THREE.MeshStandardMaterial({ color: 0x7a5636, roughness: 1 });
for (let row = 0; row < NUM_ROWS; row++) {
  const ridge = new THREE.Mesh(new THREE.BoxGeometry(12.8, 0.14, 0.42), ridgeMaterial);
  ridge.position.set(0, 0.1, row - (NUM_ROWS - 1) / 2);
  ridge.receiveShadow = true;
  scene.add(ridge);
}

// One black drip line along each row.
const pipeMaterial = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.6 });
for (let row = 0; row < NUM_ROWS; row++) {
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 12.8, 6), pipeMaterial);
  pipe.rotation.z = Math.PI / 2;
  pipe.position.set(0, 0.19, row - (NUM_ROWS - 1) / 2 + 0.22);
  scene.add(pipe);
}

// Five round huts with straw roofs, 15 to 25 metres from the field and from each other.
const hutWallMaterial = new THREE.MeshStandardMaterial({ color: 0xb48a5c, roughness: 1 });
const hutRoofMaterial = new THREE.MeshStandardMaterial({ color: 0xc9a85a, roughness: 1 });
const doorMaterial = new THREE.MeshStandardMaterial({ color: 0x3b2a1c });
const fenceMaterial = new THREE.MeshStandardMaterial({ color: 0x8d7247, roughness: 1 });

// Each hut has an x position, a z position and a direction it faces.
const HUTS = [
  [-13, -8, 0.6],
  [4, -22, 2.2],
  [22, -8, 3.6],
  [20, 14, 5],
  [-8, 20, 1.4],
];

// Build one hut with a door, a small granary and a stick fence.
function buildHut(x, z, rotation, size) {
  const hut = new THREE.Group();

  const wall = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.15, 1.3, 14), hutWallMaterial);
  wall.position.y = 0.65;
  wall.castShadow = true;
  hut.add(wall);

  const roof = new THREE.Mesh(new THREE.ConeGeometry(1.5, 1.3, 14), hutRoofMaterial);
  roof.position.y = 1.92;
  roof.castShadow = true;
  hut.add(roof);

  const roofTip = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.35, 6), fenceMaterial);
  roofTip.position.y = 2.7;
  hut.add(roofTip);

  const door = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.85, 0.06), doorMaterial);
  door.position.set(0, 0.43, 1.13);
  hut.add(door);

  // The granary stands on four short legs.
  const granary = new THREE.Group();
  const granaryWall = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.38, 0.8, 10), hutWallMaterial);
  granaryWall.position.y = 0.62;
  granary.add(granaryWall);

  const granaryRoof = new THREE.Mesh(new THREE.ConeGeometry(0.6, 0.6, 10), hutRoofMaterial);
  granaryRoof.position.y = 1.3;
  granary.add(granaryRoof);

  const legPositions = [[0.25, 0.25], [-0.25, 0.25], [0.25, -0.25], [-0.25, -0.25]];
  for (const legPosition of legPositions) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.25, 5), fenceMaterial);
    leg.position.set(legPosition[0], 0.12, legPosition[1]);
    granary.add(leg);
  }
  granary.position.set(2, 0, -0.6);
  granary.traverse(function (part) {
    part.castShadow = true;
  });
  hut.add(granary);

  // A fence of sticks around the hut, with a gap in front of the door.
  for (let i = 0; i < 26; i++) {
    const angle = (i / 26) * Math.PI * 2;
    if (Math.abs(angle - Math.PI / 2) < 0.35) {
      continue;
    }
    const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 0.55, 4), fenceMaterial);
    stick.position.set(Math.cos(angle) * 3.3, 0.27, Math.sin(angle) * 3.3);
    hut.add(stick);
  }

  hut.position.set(x, 0, z);
  hut.rotation.y = rotation;
  hut.scale.setScalar(size);
  scene.add(hut);
}

for (let i = 0; i < HUTS.length; i++) {
  // The first hut is a little smaller.
  let size = 1;
  if (i === 0) {
    size = 0.9;
  }
  buildHut(HUTS[i][0], HUTS[i][1], HUTS[i][2], size);
}

// Two dirt paths wind past the huts and both stay at least 3 metres outside the plot.
const pathMaterial = new THREE.MeshStandardMaterial({ color: 0x9a7d54, roughness: 1 });
const pathPoints = [
  [[-34, 0, -12], [-25, 0, -10], [-17, 0, -6.2], [-11.5, 0, -6], [-8.8, 0, -8.2], [-2, 0, -8.6], [4, 0, -8], [9, 0, -10], [8, 0, -15], [5, 0, -19]],
  [[40, 0, 0], [32, 0, 6], [27, 0, 12], [22, 0, 10], [16, 0, 7.5], [10.5, 0, 8.6], [5, 0, 9.6], [-2, 0, 9.4], [-8, 0, 10.2], [-11.5, 0, 14], [-8.5, 0, 17]],
];

const pathCurves = [];
for (const points of pathPoints) {
  const vectors = [];
  for (const point of points) {
    vectors.push(new THREE.Vector3(point[0], point[1], point[2]));
  }
  pathCurves.push(new THREE.CatmullRomCurve3(vectors));
}

// These squares of the ground are on or near a path, and grass and trees stay off them.
const pathSquares = new Set();

function squareKey(x, z) {
  return Math.floor(x) + "," + Math.floor(z);
}

// Cut each path into 120 small flat pieces.
for (const curve of pathCurves) {
  const pieceCount = 120;
  for (let i = 0; i < pieceCount; i++) {
    const start = curve.getPoint(i / pieceCount);
    const end = curve.getPoint((i + 1) / pieceCount);

    for (let dx = -1; dx <= 1; dx++) {
      for (let dz = -1; dz <= 1; dz++) {
        pathSquares.add(squareKey(start.x + dx, start.z + dz));
      }
    }

    const piece = new THREE.Mesh(new THREE.PlaneGeometry(1.15, start.distanceTo(end) + 0.08), pathMaterial);
    piece.rotation.order = "YXZ";
    piece.rotation.y = Math.atan2(end.x - start.x, end.z - start.z);
    piece.rotation.x = -Math.PI / 2;
    piece.position.set((start.x + end.x) / 2, 0.012, (start.z + end.z) / 2);
    piece.receiveShadow = true;
    scene.add(piece);
  }
}

function isNearPath(x, z) {
  return pathSquares.has(squareKey(x, z));
}

function isNearHut(x, z, distance = 3.9) {
  for (const hut of HUTS) {
    if (Math.hypot(hut[0] - x, hut[1] - z) < distance) {
      return true;
    }
  }
  return false;
}

function isInField(x, z, margin = 1) {
  return Math.abs(x) < 6.7 + margin && Math.abs(z) < 4.7 + margin;
}

// There are three kinds of tree, which are a flat acacia, a round broad tree and a tall slender tree.
const treeRandom = makeRandom(99);
const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x5b4430 });
const canopyMaterials = [];
for (const color of [0x56743a, 0x4a6a32, 0x68863f, 0x3f5f2e]) {
  canopyMaterials.push(new THREE.MeshStandardMaterial({ color: color, roughness: 0.9 }));
}

let treesPlaced = 0;
for (let attempt = 0; treesPlaced < 26 && attempt < 400; attempt++) {
  const angle = treeRandom() * Math.PI * 2;
  const distance = 14 + treeRandom() * 28;
  const x = Math.cos(angle) * distance;
  const z = Math.sin(angle) * distance;
  if (isNearPath(x, z) || isNearHut(x, z, 5) || isInField(x, z, 4)) {
    continue;
  }
  treesPlaced++;

  const treeType = Math.floor(treeRandom() * 3);
  const tree = new THREE.Group();
  const canopyMaterial = canopyMaterials[Math.floor(treeRandom() * canopyMaterials.length)];

  // The trunk height depends on the kind of tree.
  let trunkHeight;
  let trunkRadius;
  if (treeType === 2) {
    trunkHeight = 3 + treeRandom() * 1.5;
    trunkRadius = 0.1;
  } else if (treeType === 1) {
    trunkHeight = 1 + treeRandom() * 0.5;
    trunkRadius = 0.24;
  } else {
    trunkHeight = 1.6 + treeRandom() * 0.8;
    trunkRadius = 0.15;
  }

  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(trunkRadius * 0.7, trunkRadius, trunkHeight, 6), trunkMaterial);
  trunk.position.y = trunkHeight / 2;
  trunk.castShadow = true;
  tree.add(trunk);

  const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 9, 7), canopyMaterial);
  if (treeType === 0) {
    const widthX = 1.9 + treeRandom() * 0.7;
    const height = 0.6 + treeRandom() * 0.2;
    const widthZ = 1.9 + treeRandom() * 0.7;
    canopy.scale.set(widthX, height, widthZ);
    canopy.position.y = trunkHeight + 0.25;
  } else if (treeType === 1) {
    const radius = 1.4 + treeRandom() * 0.6;
    canopy.scale.set(radius, radius * 0.85, radius);
    canopy.position.y = trunkHeight + radius * 0.7;
  } else {
    const widthX = 0.8 + treeRandom() * 0.3;
    const height = 1.2 + treeRandom() * 0.5;
    const widthZ = 0.8 + treeRandom() * 0.3;
    canopy.scale.set(widthX, height, widthZ);
    canopy.position.y = trunkHeight + 1;
  }
  canopy.castShadow = true;
  tree.add(canopy);

  tree.position.set(x, 0, z);
  tree.rotation.y = treeRandom() * 6.28;
  tree.scale.setScalar(0.75 + treeRandom() * 0.85);
  scene.add(tree);
}

// About 25,000 grass blades grow in clumps of 6, but none grow on the field, the paths or near the huts.
const grassRandom = makeRandom(321);
const bladeGeometry = new THREE.ConeGeometry(0.025, 0.28, 3).translate(0, 0.14, 0);
const MAX_BLADES = 25200;
const grass = new THREE.InstancedMesh(bladeGeometry, new THREE.MeshStandardMaterial({ roughness: 0.9 }), MAX_BLADES);

const grassColors = [];
for (const color of [0x4a7a28, 0x5a8c30, 0x3f6e22, 0x6f8f3c, 0x7f9444]) {
  grassColors.push(new THREE.Color(color));
}

function isBlockedForGrass(x, z) {
  return isInField(x, z, 0.6) || isNearPath(x, z) || isNearHut(x, z);
}

let bladeCount = 0;
for (let attempt = 0; bladeCount < MAX_BLADES && attempt < 400000; attempt++) {
  const clumpX = (grassRandom() - 0.5) * 110;
  const clumpZ = (grassRandom() - 0.5) * 110;
  if (isBlockedForGrass(clumpX, clumpZ)) {
    continue;
  }

  // One clump has 6 blades.
  for (let blade = 0; blade < 6 && bladeCount < MAX_BLADES; blade++) {
    const x = clumpX + (grassRandom() - 0.5) * 0.35;
    const z = clumpZ + (grassRandom() - 0.5) * 0.35;
    if (isBlockedForGrass(x, z)) {
      continue;
    }

    const size = 0.6 + grassRandom() * 0.9;
    position.set(x, 0, z);
    euler.set((grassRandom() - 0.5) * 0.5, grassRandom() * 6.28, (grassRandom() - 0.5) * 0.5);
    quaternion.setFromEuler(euler);
    scaleVector.set(size, size * (0.7 + grassRandom() * 0.8), size);
    matrix.compose(position, quaternion, scaleVector);
    grass.setMatrixAt(bladeCount, matrix);
    grass.setColorAt(bladeCount, grassColors[Math.floor(grassRandom() * grassColors.length)]);
    bladeCount++;
  }
}
grass.count = bladeCount;
grass.receiveShadow = true;
scene.add(grass);

// About 40 low bushes.
const bushMaterial = new THREE.MeshStandardMaterial({ color: 0x3f6a28, roughness: 0.95 });
let bushesPlaced = 0;
for (let attempt = 0; bushesPlaced < 40 && attempt < 600; attempt++) {
  const angle = grassRandom() * 6.28;
  const distance = 8 + grassRandom() * 36;
  const x = Math.cos(angle) * distance;
  const z = Math.sin(angle) * distance;
  if (isNearPath(x, z) || isNearHut(x, z) || isInField(x, z, 1.5)) {
    continue;
  }
  bushesPlaced++;

  const bush = new THREE.Mesh(new THREE.IcosahedronGeometry(0.5 + grassRandom() * 0.5, 0), bushMaterial);
  bush.position.set(x, 0.3, z);
  bush.scale.y = 0.7;
  bush.castShadow = true;
  scene.add(bush);
}

// Where each plant stands.
const plantPositions = [];
for (let i = 0; i < PLANT_COUNT; i++) {
  const column = i % NUM_COLUMNS;
  const row = Math.floor(i / NUM_COLUMNS);
  plantPositions.push(new THREE.Vector3(column - (NUM_COLUMNS - 1) / 2, 0.17, row - (NUM_ROWS - 1) / 2));
}

// A leaf is a thin strip that is bent and narrows towards its tip.
const leafGeometry = new THREE.PlaneGeometry(0.11, 0.75, 1, 6);
leafGeometry.translate(0, 0.375, 0);
const leafPoints = leafGeometry.attributes.position;
for (let i = 0; i < leafPoints.count; i++) {
  const y = leafPoints.getY(i);
  leafPoints.setZ(i, -0.38 * Math.pow(y / 0.75, 2));
  leafPoints.setX(i, leafPoints.getX(i) * (1 - Math.pow(y / 0.75, 3) * 0.85));
}
leafGeometry.computeVertexNormals();

// Each plant is made of many copies of a few shapes, which is fast to draw.
const LEAVES_PER_PLANT = 7;
const leaves = new THREE.InstancedMesh(
  leafGeometry,
  new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.75 }),
  PLANT_COUNT * LEAVES_PER_PLANT
);
leaves.castShadow = true;
scene.add(leaves);

const stalks = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(0.03, 0.045, 1, 6).translate(0, 0.5, 0),
  new THREE.MeshStandardMaterial({ roughness: 0.8 }),
  PLANT_COUNT
);
stalks.castShadow = true;
scene.add(stalks);

const tassels = new THREE.InstancedMesh(
  new THREE.ConeGeometry(0.07, 0.32, 5).translate(0, 0.16, 0),
  new THREE.MeshStandardMaterial({ color: 0xe2c46a }),
  PLANT_COUNT
);
scene.add(tassels);

const cobs = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(0.065, 0.05, 0.4, 7),
  new THREE.MeshStandardMaterial({ roughness: 0.8 }),
  PLANT_COUNT
);
cobs.castShadow = true;
scene.add(cobs);

// A dark hole where a plant was removed.
const holes = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(0.16, 0.2, 0.06, 8),
  new THREE.MeshStandardMaterial({ color: 0x2a1a0e }),
  PLANT_COUNT
);
scene.add(holes);

// A yellow ring on the ground around a plant the detector flagged.
const rings = new THREE.InstancedMesh(
  new THREE.RingGeometry(0.36, 0.44, 24).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0xffc83a, transparent: true, opacity: 0.95, side: THREE.DoubleSide }),
  PLANT_COUNT
);
scene.add(rings);

// Small light green tufts of weeds grow between the rows. They grow with the weed level and shrink when the farmer weeds.
const WEED_GAPS = NUM_ROWS + 1;
const TUFTS_PER_GAP = 24;
const BLADES_PER_TUFT = 3;
const weedRandom = makeRandom(555);
const weedSpots = [];
for (let gap = 0; gap < WEED_GAPS; gap++) {
  for (let k = 0; k < TUFTS_PER_GAP; k++) {
    weedSpots.push({
      x: (weedRandom() - 0.5) * 12.2,
      z: gap - (NUM_ROWS - 1) / 2 - 0.5 + (weedRandom() - 0.5) * 0.2,
      limit: weedRandom(),
      turn: weedRandom() * 6.28,
    });
  }
}
const weedBlades = new THREE.InstancedMesh(
  new THREE.ConeGeometry(0.045, 1, 4).translate(0, 0.5, 0),
  new THREE.MeshStandardMaterial({ color: 0x9fd04a, roughness: 0.9 }),
  weedSpots.length * BLADES_PER_TUFT
);
scene.add(weedBlades);
let shownWeedLevel = -1;

// Place every tuft for this weed level, which is a number from 0 to 1.
function drawWeeds(level) {
  if (Math.abs(level - shownWeedLevel) < 0.002) {
    return;
  }
  shownWeedLevel = level;
  let index = 0;
  for (const spot of weedSpots) {
    const size = clamp((level * 1.4 - spot.limit * 0.9) / 0.5);
    for (let blade = 0; blade < BLADES_PER_TUFT; blade++) {
      if (size <= 0) {
        weedBlades.setMatrixAt(index, hiddenMatrix);
      } else {
        const angle = spot.turn + blade * 2.1;
        position.set(spot.x + Math.cos(angle) * 0.06, 0.08, spot.z + Math.sin(angle) * 0.06);
        euler.set(Math.sin(angle) * 0.35, angle, Math.cos(angle) * 0.35);
        quaternion.setFromEuler(euler);
        const height = (0.1 + 0.32 * size) * (0.8 + 0.2 * blade);
        scaleVector.set(0.6 + 0.5 * size, height, 0.6 + 0.5 * size);
        matrix.compose(position, quaternion, scaleVector);
        weedBlades.setMatrixAt(index, matrix);
      }
      index++;
    }
  }
  weedBlades.instanceMatrix.needsUpdate = true;
}

const drone = new THREE.Group();
const droneBodyMaterial = new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.3, roughness: 0.4 });
const droneDarkMaterial = new THREE.MeshStandardMaterial({ color: 0x222222 });
drone.add(new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.14, 0.5), droneBodyMaterial));

// Four arms, each with a spinning rotor.
const armCorners = [[0.4, 0.4], [-0.4, 0.4], [0.4, -0.4], [-0.4, -0.4]];
for (const corner of armCorners) {
  const cornerX = corner[0];
  const cornerZ = corner[1];

  const arm = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.04, 0.6), droneDarkMaterial);
  arm.position.set(cornerX / 2, 0, cornerZ / 2);
  arm.lookAt(cornerX, 0, cornerZ);
  drone.add(arm);

  const rotorMaterial = new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.35 });
  const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.015, 16), rotorMaterial);
  rotor.position.set(cornerX, 0.08, cornerZ);
  drone.add(rotor);
}

const droneCamera = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 8), droneDarkMaterial);
droneCamera.position.y = -0.1;
drone.add(droneCamera);

drone.traverse(function (part) {
  part.castShadow = true;
});
scene.add(drone);

// The blue cone shows what the drone is scanning.
const scanCone = new THREE.Mesh(
  new THREE.ConeGeometry(0.9, 2.6, 24, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x7fe0ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
);
scene.add(scanCone);

// The drone waits next to the first hut and flies over the first and the last row of the field.
const REST_POSITION = new THREE.Vector3(HUTS[0][0] + 1.9, 2.3, HUTS[0][1] + 2.2);
const SWEEP_START = new THREE.Vector3(-6.2, 2.5, -3.5);
const SWEEP_END = new THREE.Vector3(-6.2, 2.5, 3.5);

// Make a cloud of small points that we can move around.
function makePoints(count, color, size, opacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const material = new THREE.PointsMaterial({ color: color, size: size, transparent: true, opacity: opacity, depthWrite: false });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  scene.add(points);
  return points;
}

// Rain is drawn as short lines.
const RAIN_COUNT = 3000;
const rainGeometry = new THREE.BufferGeometry();
rainGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RAIN_COUNT * 6), 3));
const rainLines = new THREE.LineSegments(
  rainGeometry,
  new THREE.LineBasicMaterial({ color: 0xbcd9f0, transparent: true, opacity: 0.55 })
);
rainLines.frustumCulled = false;
scene.add(rainLines);

// Spray mist and fertilizer granules are few and small, so they look like falling granules.
const sprayMist = makePoints(750, 0xf4f8ff, 0.065, 0.8);
const irrigationWater = makePoints(1500, 0x58b4ff, 0.07, 0.85);
const fertilizerGranules = makePoints(450, 0xffffff, 0.04, 0.95);

// At the start nothing is shown and the first frame places everything.
const allPlantParts = [leaves, stalks, tassels, cobs, holes, rings];
for (const part of allPlantParts) {
  for (let i = 0; i < part.count; i++) {
    part.setMatrixAt(i, hiddenMatrix);
  }
}

// Colours must exist before the first render, so we give every instance white now.
const white = new THREE.Color(0xffffff);
for (const part of [leaves, stalks, cobs]) {
  for (let i = 0; i < part.count; i++) {
    part.setColorAt(i, white);
  }
}

// Four fixed random numbers for each particle, so the particles do not jump around.
const particleSeeds = [];
for (let i = 0; i < 6000; i++) {
  const random = makeRandom(i * 7 + 3);
  particleSeeds.push([random(), random(), random(), random()]);
}

// Move every point of a cloud to the position given by positionFunction, or hide the cloud.
function movePoints(points, isOn, positionFunction) {
  const coordinates = points.geometry.attributes.position.array;
  const pointCount = coordinates.length / 3;
  for (let i = 0; i < pointCount; i++) {
    if (isOn) {
      coordinates.set(positionFunction(i, particleSeeds[i]), i * 3);
    } else {
      coordinates.set([0, -50, 0], i * 3);
    }
  }
  points.geometry.attributes.position.needsUpdate = true;
}

// These colours show a healthy plant, water stress, blight lesions, severe blight and a mature plant.
const healthyGreen = new THREE.Color(0x3f8f37);
const stressGreen = new THREE.Color(0x8fb8a8);
const lesionColor = new THREE.Color(0xa59a7e);
const brownColor = new THREE.Color(0x6b3a1c);
const goldColor = new THREE.Color(0xe0aa30);
const stalkGreen = new THREE.Color(0x5e8f3a);
const huskGreen = new THREE.Color(0x9cbf5a);
const daySky = new THREE.Color(0x9fc6e4);
const rainSky = new THREE.Color(0x5d6f7c);

function drawPlant(plantIndex, weekProgress, severity, biomass, isRemoved, removeProgress, moisture) {
  const spot = plantPositions[plantIndex];

  // A removed plant is gone and we only see a dark hole.
  if (isRemoved && removeProgress >= 1) {
    for (let leaf = 0; leaf < LEAVES_PER_PLANT; leaf++) {
      leaves.setMatrixAt(plantIndex * LEAVES_PER_PLANT + leaf, hiddenMatrix);
    }
    stalks.setMatrixAt(plantIndex, hiddenMatrix);
    tassels.setMatrixAt(plantIndex, hiddenMatrix);
    cobs.setMatrixAt(plantIndex, hiddenMatrix);
    matrix.makeTranslation(spot.x, 0.17, spot.z);
    holes.setMatrixAt(plantIndex, matrix);
    return;
  }
  holes.setMatrixAt(plantIndex, hiddenMatrix);

  // A plant that is being pulled out lifts up.
  let lift = 0;
  if (isRemoved) {
    lift = removeProgress;
  }

  const growth = smooth(Math.min(1, (weekProgress + 0.6) / 12));

  // How much the plant has grown, compared with a healthy plant.
  let expectedBiomass = 0;
  for (let i = 0; i < Math.ceil(weekProgress); i++) {
    expectedBiomass += STAGE_WEIGHTS[i];
  }
  if (expectedBiomass === 0) {
    expectedBiomass = 0.01;
  }
  const vigor = clamp(0.55 + (0.45 * biomass) / (expectedBiomass * 0.8), 0.4, 1);

  // An infected plant stays shorter.
  const infection = clamp(severity / 0.6, 0, 1);
  const plantHeight = (0.18 + 2.1 * growth) * vigor * (1 - 0.4 * infection);

  let waterStress = 0;
  if (moisture < 0.3) {
    waterStress = 1 - moisture / 0.3;
  }
  const ripeness = smooth((weekProgress - 14.5) / 3);
  // A plant turns brown only when the blight is severe.
  const brownness = smooth((severity - 0.55) / 0.35);

  const jitter = ((plantIndex * 37) % 11) / 11;
  const tiltOut = lift * 1.4;

  // A mature plant is golden when healthy and brown when infected.
  matureColor.copy(goldColor).lerp(brownColor, infection * 0.75);

  // Draw the stalk.
  partColor.copy(stalkGreen).lerp(matureColor, ripeness * 0.8).lerp(brownColor, brownness);
  position.set(spot.x, spot.y + lift * 1.2, spot.z);
  euler.set(tiltOut * 0.3, 0, tiltOut * 0.4);
  quaternion.setFromEuler(euler);
  scaleVector.set(1, plantHeight, 1);
  matrix.compose(position, quaternion, scaleVector);
  stalks.setMatrixAt(plantIndex, matrix);
  stalks.setColorAt(plantIndex, partColor);

  // Draw the leaves, where a young plant has fewer leaves.
  const leafCount = Math.max(2, Math.min(LEAVES_PER_PLANT, Math.round(2 + growth * LEAVES_PER_PLANT)));
  for (let leaf = 0; leaf < LEAVES_PER_PLANT; leaf++) {
    const instanceIndex = plantIndex * LEAVES_PER_PLANT + leaf;
    if (leaf >= leafCount) {
      leaves.setMatrixAt(instanceIndex, hiddenMatrix);
      continue;
    }

    const heightOnStalk = 0.12 + (0.75 * leaf) / LEAVES_PER_PLANT;
    const turn = leaf * 2.4 + jitter * 6;
    const droop = 0.15 * severity + 0.3 * waterStress + 0.35 * ripeness;
    const tilt = 0.55 + 0.35 * (1 - leaf / LEAVES_PER_PLANT) + droop + tiltOut;
    const leafSize = (0.35 + 0.75 * growth) * (1 - (0.3 * leaf) / LEAVES_PER_PLANT) * vigor;

    position.set(spot.x, spot.y + plantHeight * heightOnStalk + lift * 1.2, spot.z);
    euler.set(tilt, turn, 0, "YXZ");
    quaternion.setFromEuler(euler);
    scaleVector.set(leafSize * 1.2, leafSize, leafSize);
    matrix.compose(position, quaternion, scaleVector);
    leaves.setMatrixAt(instanceIndex, matrix);

    // The lower leaves get lesions first.
    const lesion = clamp(severity * 1.8 - (leaf / LEAVES_PER_PLANT) * 0.8, 0, 1) * 0.9;
    leafColor.copy(healthyGreen);
    leafColor.lerp(stressGreen, waterStress * 0.9);
    leafColor.lerp(matureColor, ripeness * 0.9);
    leafColor.lerp(lesionColor, lesion);
    leafColor.lerp(brownColor, brownness);
    leaves.setColorAt(instanceIndex, leafColor);
  }

  // The tassel appears around week 9.
  const tasselSize = smooth((weekProgress - 8.5) / 1.5);
  position.set(spot.x, spot.y + plantHeight + lift * 1.2, spot.z);
  quaternion.identity();
  scaleVector.set(tasselSize, tasselSize, tasselSize);
  matrix.compose(position, quaternion, scaleVector);
  tassels.setMatrixAt(plantIndex, matrix);

  // The cob appears around week 11 and is smaller on an infected plant.
  const cobSize = smooth((weekProgress - 10.2) / 1.3) * (1 - 0.55 * infection);
  position.set(spot.x + 0.1, spot.y + plantHeight * 0.5 + lift * 1.2, spot.z + 0.02);
  euler.set(0, 0, -0.55);
  quaternion.setFromEuler(euler);
  scaleVector.set(cobSize, cobSize, cobSize);
  matrix.compose(position, quaternion, scaleVector);
  cobs.setMatrixAt(plantIndex, matrix);
  partColor.copy(huskGreen).lerp(matureColor, clamp((weekProgress - 12) / 5)).lerp(brownColor, brownness * 0.8);
  cobs.setColorAt(plantIndex, partColor);
}

// Find the place along the flight, from 0 to 1, where the drone reaches this plant, because it flies row by row.
function sweepPlace(plantIndex) {
  const row = Math.floor(plantIndex / NUM_COLUMNS);
  const column = plantIndex % NUM_COLUMNS;

  // Even rows go left to right, odd rows go right to left.
  let placeInRow = (column + 0.5) / NUM_COLUMNS;
  if (row % 2 === 1) {
    placeInRow = 1 - (column + 0.5) / NUM_COLUMNS;
  }
  return (row + placeInRow) / NUM_ROWS;
}

// Drone scout, spray and fertilize send the drone over the field.
function isFlyingAction(action) {
  return action === 1 || action === 2 || action === 4 || action === 6;
}

// The camera only moves when the user moves it, and it turns around the middle of the field.
const CAMERA_TARGET = new THREE.Vector3(-2.5, 0.3, -1.2);
const HOME_DISTANCE = Math.hypot(5.5, 9.4, 20);
const HOME_TURN = Math.atan2(20, 5.5);
const HOME_HEIGHT = Math.asin(9.4 / HOME_DISTANCE);
let cameraTurn = HOME_TURN;
let cameraHeight = HOME_HEIGHT;
let cameraDistance = HOME_DISTANCE;

function placeCamera() {
  const flat = Math.cos(cameraHeight);
  camera.position.set(
    CAMERA_TARGET.x + Math.cos(cameraTurn) * flat * cameraDistance,
    CAMERA_TARGET.y + Math.sin(cameraHeight) * cameraDistance,
    CAMERA_TARGET.z + Math.sin(cameraTurn) * flat * cameraDistance
  );
  camera.lookAt(CAMERA_TARGET);
}

function drawFrame(frameNumber) {
  if (!startState) {
    return;
  }

  const place = whereAmI(frameNumber);
  const time = place.time;
  const weekIndex = place.weekIndex;
  const phase = place.phase;

  let step = null;
  if (weekIndex >= 0) {
    step = steps[weekIndex];
  }

  // A week has four parts, which are looking at the field, the action, the weather and the growth, and they follow each other.
  let growthTime = 0;
  let weekProgress = 0;
  let moisture = startState.moisture;
  let actionTime = -1;
  let plantsBefore = startState.plants;
  let plantsAfter = startState.plants;
  if (step) {
    growthTime = smooth((phase - 0.68) / 0.3);
    weekProgress = step.week + growthTime;
    moisture = lerp(step.moistureBefore, step.moistureAfter, smooth((phase - 0.5) / 0.3));
    actionTime = (phase - 0.12) / 0.43;
    plantsBefore = step.before;
    plantsAfter = step.after;
  }

  // Draw every plant.
  for (let i = 0; i < PLANT_COUNT; i++) {
    const severity = lerp(plantsBefore.sev[i], plantsAfter.sev[i], growthTime);
    const biomass = lerp(plantsBefore.bio[i], plantsAfter.bio[i], growthTime);

    let isRemoved = plantsBefore.removed[i];
    let removeProgress = 0;
    if (isRemoved) {
      removeProgress = 1;
    }
    // A plant removed this week is pulled out during the action phase.
    if (step && !plantsBefore.removed[i] && plantsAfter.removed[i]) {
      isRemoved = true;
      removeProgress = smooth((actionTime - 0.45) / 0.5);
    }
    drawPlant(i, weekProgress, severity, biomass, isRemoved, removeProgress, moisture);
  }
  for (const part of [leaves, stalks, tassels, cobs, holes]) {
    part.instanceMatrix.needsUpdate = true;
    if (part.instanceColor) {
      part.instanceColor.needsUpdate = true;
    }
  }

  // The drone drops a yellow ring when it passes a flagged plant, and the ring stays until the plant is removed or the next drone scout.
  for (let i = 0; i < PLANT_COUNT; i++) {
    let showRing = false;
    if (step) {
      if (step.action === 1 || step.action === 6) {
        showRing = step.flagsAfterAction[i] && clamp(actionTime) >= sweepPlace(i) && !(step.action === 6 && actionTime > 0.9);
      } else if (step.action === 5) {
        showRing = step.flagsAfterAction[i] && actionTime < 0.45;
      } else {
        showRing = plantsBefore.flag[i] && !plantsAfter.removed[i];
      }
    }

    let ringSize = 0;
    if (showRing) {
      ringSize = 1;
    }
    position.set(plantPositions[i].x, 0.24, plantPositions[i].z);
    quaternion.identity();
    scaleVector.set(ringSize, ringSize, ringSize);
    matrix.compose(position, quaternion, scaleVector);
    rings.setMatrixAt(i, matrix);
  }
  rings.instanceMatrix.needsUpdate = true;

  // The weeds grow during the week, and weeding clears them during the action.
  let weedLevel = startState.obs[10] || 0;
  if (step) {
    if (step.action === 7) {
      weedLevel = lerp(step.weedsBefore * (1 - smooth(actionTime)), step.weedsAfter, growthTime);
    } else {
      weedLevel = lerp(step.weedsBefore, step.weedsAfter, growthTime);
    }
  }
  drawWeeds(weedLevel);

  // The drone waits by the hut and for drone scout, spray and fertilize it flies over the field row by row, just above the plants, and then flies home.
  let sweeping = false;
  if (step && isFlyingAction(step.action)) {
    const flyHeight = 0.18 + 2.1 * smooth(Math.min(1, (step.week + 0.6) / 12)) + 0.8;
    const flightStart = SWEEP_START.clone().setY(flyHeight);
    const flightEnd = SWEEP_END.clone().setY(flyHeight);

    if (phase < 0.12) {
      drone.position.lerpVectors(REST_POSITION, flightStart, smooth(phase / 0.12));
    } else if (actionTime <= 1) {
      sweeping = true;

      // Work out which row the drone is over and how far along that row it is.
      const progress = clamp(actionTime) * NUM_ROWS;
      const row = Math.min(NUM_ROWS - 1, Math.floor(progress));
      const alongRow = progress - row;

      // Even rows go left to right, odd rows go right to left.
      let direction = 1;
      if (row % 2 === 1) {
        direction = -1;
      }

      const x = direction * (alongRow - 0.5) * 12.4;
      const y = flyHeight + Math.sin(frameNumber * 0.3) * 0.04;
      const z = row - (NUM_ROWS - 1) / 2;
      drone.position.set(x, y, z);
    } else if (phase < 0.7) {
      drone.position.lerpVectors(flightEnd, REST_POSITION, smooth((phase - 0.55) / 0.15));
    } else {
      drone.position.copy(REST_POSITION);
    }
  } else {
    drone.position.set(REST_POSITION.x, REST_POSITION.y + Math.sin(frameNumber * 0.1) * 0.05, REST_POSITION.z);
  }

  // The scan cone is only visible while the drone scouts.
  scanCone.visible = sweeping && (step.action === 1 || step.action === 6);
  scanCone.scale.set(1, Math.max(0.3, drone.position.y / 2.6), 1);
  scanCone.position.set(drone.position.x, drone.position.y / 2, drone.position.z);

  // The rain falls in the middle of the week, and more rain means more streaks.
  let rainAmount = 0;
  if (step) {
    rainAmount = step.rain * smooth((phase - 0.5) / 0.08) * (1 - smooth((phase - 0.78) / 0.08));
  }
  const rainStreaks = Math.min(RAIN_COUNT, Math.round(rainAmount * 30));
  const rainCoordinates = rainGeometry.attributes.position.array;
  for (let i = 0; i < RAIN_COUNT; i++) {
    if (i < rainStreaks) {
      const seed = particleSeeds[i];
      const y = (((seed[1] * 12 - time * 14) % 12) + 12) % 12;
      const x = seed[0] * 40 - 20;
      const z = seed[2] * 34 - 14;
      rainCoordinates.set([x, y, z, x - 0.05, y + 0.45, z], i * 6);
    } else {
      rainCoordinates.fill(0, i * 6, i * 6 + 6);
    }
  }
  rainGeometry.attributes.position.needsUpdate = true;

  // The sky turns grey and the light gets weaker when it rains.
  const sky = daySky.clone().lerp(rainSky, clamp(rainAmount / 70));
  scene.background = sky;
  scene.fog.color.copy(sky);
  sun.intensity = 0.85 - 0.4 * clamp(rainAmount / 70);

  // Spray mist falls from the drone onto the row below it.
  function mistPosition(index, seed) {
    const fall = ((seed[1] + time * 0.9) % 1) * (drone.position.y - 0.3);
    const x = drone.position.x + (seed[0] - 0.5) * 1.3;
    const y = drone.position.y - 0.2 - fall;
    const z = drone.position.z + (seed[2] - 0.5) * 0.4;
    return [x, y, z];
  }
  movePoints(sprayMist, sweeping && step.action === 2, mistPosition);

  // Irrigation water moves along the drip lines.
  function waterPosition(index, seed) {
    const row = Math.floor(seed[0] * NUM_ROWS);
    const x = seed[1] * 12.6 - 6.3;
    const life = (seed[2] + time * 1.5) % 1;
    const y = 0.22 + Math.sin(life * Math.PI) * 0.25;
    const z = row - (NUM_ROWS - 1) / 2 + 0.22 - life * 0.3;
    return [x, y, z];
  }
  const isIrrigating = step && step.action === 3 && actionTime > 0 && actionTime < 1.1;
  movePoints(irrigationWater, isIrrigating, waterPosition);

  // Fertilizer granules fall from the drone onto the row below it.
  function granulePosition(index, seed) {
    const fall = ((seed[1] + time * 1.4) % 1) * (drone.position.y - 0.3);
    const x = drone.position.x + (seed[0] - 0.5) * 1.1;
    const y = drone.position.y - 0.2 - fall;
    const z = drone.position.z + (seed[2] - 0.5) * 0.35;
    return [x, y, z];
  }
  movePoints(fertilizerGranules, sweeping && step.action === 4, granulePosition);

  // Wetter soil is darker.
  plotMaterial.color.setRGB(0.54 - 0.2 * moisture, 0.39 - 0.15 * moisture, 0.25 - 0.1 * moisture);

  placeCamera();
  renderer.render(scene, camera);

  updateHud(step, place.isOutro, weekIndex);
}

// The plain words that the page shows for the code names of the 12 numbers.
const PLAIN_NAMES = {
  week: "Week",
  growth_stage: "Growth stage",
  soil_moisture: "Soil water",
  rain_forecast: "Rain forecast",
  nitrogen: "Soil food (nitrogen)",
  detected_disease: "Sick plants found",
  weeks_since_scout: "Time since drone flight",
  fungicide_left: "Spray protection left",
  budget: "Money left",
  plants_removed: "Plants removed",
  weeds: "Weeds",
  sprays_used: "Sprays used (of 3)",
};

// The plain words that the page shows for the growth stages.
const PLAIN_STAGES = {
  Emergence: "Sprouting",
  Vegetative: "Leaf growth",
  Flowering: "Flowering",
  "Grain fill": "Cobs filling",
  Maturity: "Ripening",
};

// Show one line for each of the 12 numbers that the farmer sees, with a small bar.
function observationRows(list) {
  let html = "";
  for (const item of list) {
    html += '<div class="o">';
    html += "<span>" + (PLAIN_NAMES[item[0]] || item[0]) + "</span>";
    html += '<b><i style="width:' + clamp(item[1]) * 100 + '%"></i></b>';
    html += "<span>" + item[1].toFixed(2) + "</span>";
    html += "</div>";
  }
  return html;
}

// Write an amount of money with a minus sign when it is negative.
function money(value) {
  const rounded = Math.round(value);
  let sign = "";
  if (rounded < 0) {
    sign = "−";
  }
  return sign + formatNumber(Math.abs(rounded)) + " XAF";
}

// Find one of the 12 numbers that the farmer sees, by its name.
function observationValue(list, name) {
  for (const item of list) {
    if (item[0] === name) {
      return item[1];
    }
  }
  return 0;
}

// Say who chooses the actions, which is the user in the I play mode.
function whoDecides() {
  if (mode === "play") {
    return "You";
  }
  return farmerTitle();
}

// The name of the farmer, where a trained farmer is called for example the PPO farmer.
function farmerTitle() {
  if (trainedFarmer) {
    return seasonLabel + " farmer";
  }
  return seasonLabel;
}

// The name of the farmer inside a sentence, for example the PPO farmer.
function farmerPhrase() {
  if (trainedFarmer) {
    return "the " + seasonLabel + " farmer";
  }
  return seasonLabel;
}

// The same as farmerPhrase, with a capital letter, for the start of a sentence.
function farmerSentence() {
  const phrase = farmerPhrase();
  return phrase.charAt(0).toUpperCase() + phrase.slice(1);
}

// Write the penalty of a failed crop, for example −10 points.
function penaltyText(penalty) {
  return "−" + Number(penalty.toFixed(2)) + " points";
}

// Explain the penalty under the table, but only when a crop failed.
function penaltyNote(penalty) {
  return '<p class="note">The crop failed this season, so a penalty of ' + Number(penalty.toFixed(2)) + " points was taken away from the score, although this penalty is not money paid by the farmer.</p>";
}

// Build the card that shows the result of a season that the trained farmer played.
function watchCardHtml(last, spent, sold) {
  let html = '<div class="acc">';
  html += "<span>Starting money</span><span>" + money(config.budget) + "</span>";
  html += "<span>Spent</span><span>" + money(-spent) + "</span>";
  html += "<span>Grain sold (kg × " + config.price + ")</span><span>" + money(sold) + "</span>";
  html += '<span class="tot">Final money</span><span class="tot">' + money(last.budgetAfter + sold) + "</span>";
  html += "<span>Profit</span><span>" + money(sold - spent) + "</span>";
  if (failurePenalty !== null) {
    html += "<span>Crop failure penalty</span><span>" + penaltyText(failurePenalty) + "</span>";
  }
  html += "<span>Score</span><span>" + last.totalReturn.toFixed(2) + "</span>";
  html += "<span>Good decisions (judge)</span><span>" + last.good + " of " + last.judged + "</span>";
  html += "</div>";
  if (failurePenalty !== null) {
    html += penaltyNote(failurePenalty);
  }
  return html;
}

// Build the card that compares the season of the user with the season of the trained farmer.
function playCardHtml(last, spent, sold) {
  const other = comparison;
  const otherSold = other.kg * config.price;
  let html = '<div class="acc cmp">';
  html += '<span></span><span class="h">You</span><span class="h">' + farmerTitle() + "</span>";
  html += "<span>Harvest</span><span>" + formatNumber(last.kg) + " kg</span><span>" + formatNumber(other.kg) + " kg</span>";
  html += "<span>Spent</span><span>" + money(-spent) + "</span><span>" + money(-other.spent) + "</span>";
  html += "<span>Grain sold</span><span>" + money(sold) + "</span><span>" + money(otherSold) + "</span>";
  html += '<span class="tot">Final money</span><span class="tot">' + money(config.budget - spent + sold) + '</span><span class="tot">' + money(config.budget - other.spent + otherSold) + "</span>";
  html += "<span>Profit</span><span>" + money(sold - spent) + "</span><span>" + money(otherSold - other.spent) + "</span>";
  const failed = failurePenalty !== null || other.failure_penalty !== null;
  if (failed) {
    const yours = failurePenalty !== null ? penaltyText(failurePenalty) : "";
    const theirs = other.failure_penalty !== null ? penaltyText(other.failure_penalty) : "";
    html += "<span>Crop failure penalty</span><span>" + yours + "</span><span>" + theirs + "</span>";
  }
  html += "<span>Score</span><span>" + last.totalReturn.toFixed(2) + "</span><span>" + other.ret.toFixed(2) + "</span>";
  html += "<span>Good decisions (judge)</span><span>" + last.good + " of " + last.judged + "</span><span>" + other.good + " of " + other.judged + "</span>";
  html += "</div>";
  if (failed) {
    const shown = failurePenalty !== null ? failurePenalty : other.failure_penalty;
    html += penaltyNote(shown);
  }

  let verdict;
  if (last.totalReturn >= other.ret) {
    verdict = "You did as well as " + farmerPhrase() + " or better on this season.";
  } else {
    verdict = farmerSentence() + " earned " + money((other.ret - last.totalReturn) * 10000) + " more than you on this season.";
  }
  html += '<p class="verdict">' + verdict + "</p>";
  return html;
}

// Show the card at the end of the season, or hide it.
function showCard(isOutro) {
  if (!isOutro) {
    $("card").style.display = "none";
    $("dim").style.display = "none";
    return;
  }

  if (steps.length === 0 || (mode === "play" && !comparison)) {
    $("card").style.display = "none";
    $("dim").style.display = "none";
    return;
  }

  const last = steps[steps.length - 1];
  const spent = config.budget - last.budgetAfter;
  const sold = last.kg * config.price;
  let title = "CROP FAILURE";
  if (last.end === "Harvest") {
    title = "SEASON COMPLETE";
  }

  let who = seasonLabel;
  if (mode === "play") {
    who = "Your season";
  }
  who += " · " + formatNumber(last.kg) + " kg (" + (last.kg / (config.plot_ha * 1000)).toFixed(2) + " t/ha)";

  let body = watchCardHtml(last, spent, sold);
  let againText = "REPLAY";
  if (mode === "play") {
    body = playCardHtml(last, spent, sold);
    againText = "PLAY AGAIN";
  }

  let html = '<div style="text-align:center;font-size:30px">🌽</div>';
  html += "<h2>" + title + '</h2><div class="who">' + who + "</div>" + body;
  html += '<div class="btns"><button class="rp" id="again">' + againText + '</button><button class="cl" id="close">CLOSE</button></div>';
  $("card").innerHTML = html;
  $("card").style.display = "block";
  $("dim").style.display = "block";
  $("again").onclick = startSeason;
  $("close").onclick = function () {
    $("card").style.display = "none";
    $("dim").style.display = "none";
  };
}

// Show the notes, the chances and the judge in the panel on the left.
function updateNotesPanel(step, atWeekEnd) {
  // While a week moves we show what the farmer saw, and when it is over we show what the farmer sees now.
  let notes = nextNotes;
  let chances = nextChances;
  if (step && (seasonDone || !atWeekEnd)) {
    notes = step.notes;
    chances = step.chances;
  }

  let html = "";
  for (const note of notes) {
    html += "<li>" + note + "</li>";
  }
  $("nNotes").innerHTML = html;

  // The top 3 actions on one line, and the first one is bold and coloured.
  $("nChanceBox").style.display = chances ? "block" : "none";
  if (chances) {
    if (chanceKind === "preference") {
      $("nChanceTitle").textContent = seasonLabel + " farmer prefers";
    } else {
      $("nChanceTitle").textContent = seasonLabel + " farmer is sure of";
    }
    const order = chances.map(function (value, index) { return index; }).sort(function (a, b) { return chances[b] - chances[a]; });
    const parts = [];
    for (const index of order.slice(0, 3)) {
      const text = SHORT_NAMES[index] + " " + Math.round(chances[index] * 100) + "%";
      parts.push(parts.length === 0 ? "<b>" + text + "</b>" : text);
    }
    $("nChances").innerHTML = parts.join(" · ");
  }

  // The judge looks at the last choice, and the count of good decisions is in the tile at the top.
  let judgeHtml = '<div class="sofar">No choice to judge yet.</div>';
  if (step && step.judge) {
    const verdict = step.judge;
    const names = { good: "Good decision", ok: "Could be better", poor: "Poor decision" };
    judgeHtml = '<div class="jv lv-' + verdict.level + '">' + names[verdict.level] + ": " + step.actionName + "</div>";
    judgeHtml += "<div>" + verdict.text + "</div>";
    if (verdict.better !== null && verdict.better !== undefined) {
      judgeHtml += '<div class="better">Better: <b>' + config.actions[verdict.better].name + "</b>, because " + verdict.better_why + "</div>";
    } else if (mode === "play") {
      judgeHtml += '<div class="would">' + farmerSentence() + " would have chosen <b>" + config.actions[step.modelPick].name + "</b>.</div>";
    }
  }
  $("nJudge").innerHTML = judgeHtml;
}

// Update all the panels, but only when something changed, so the page stays fast.
function updateHud(step, isOutro, weekIndex) {
  const atWeekEnd = frame >= lastFrameAvailable() - 1;
  const key = [weekIndex, isOutro, atWeekEnd, mode, busy, steps.length, advice, seasonLabel, errorText].join();
  if (key === lastHudKey) {
    return;
  }
  lastHudKey = key;
  $("vPol").textContent = whoDecides();

  // The user is choosing when it is the I play mode and nothing is moving.
  const choosing = mode === "play" && !busy && !seasonDone;

  // Before the first week.
  if (!step) {
    $("vWeek").textContent = "0 / " + WEEKS;
    $("vStage").textContent = "Planting";
    $("vGood").textContent = "0 / 0";
    if (choosing) {
      $("vAct").textContent = "CHOOSE YOUR ACTION FOR WEEK 1";
    } else if (busy) {
      $("vAct").textContent = "SEASON STARTS";
    } else {
      $("vAct").textContent = "PRESS NEXT WEEK OR AUTO RUN";
    }
    if (errorText) {
      $("vAct").textContent = errorText;
    }
    $("vBudget").textContent = money(startState.budget);
    $("vDet").textContent = Math.round(observationValue(nextObs, "detected_disease") * 100) + "%";
    $("vWr").textContent = money(0);
    $("vWr").className = "val";
    $("vWrs").textContent = "score 0.00";
    $("vTr").textContent = money(0);
    $("vTrs").textContent = "score 0.00";
    $("obs").innerHTML = observationRows(nextObs);
    updateNotesPanel(null, true);
    showCard(false);
    return;
  }

  $("vWeek").textContent = step.week + 1 + " / " + WEEKS;
  $("vStage").textContent = PLAIN_STAGES[step.stage] || step.stage;
  $("vGood").textContent = step.good + " / " + step.judged;

  if (choosing) {
    $("vAct").textContent = "CHOOSE YOUR ACTION FOR WEEK " + (steps.length + 1);
  } else {
    let text = step.actionName.toUpperCase();
    if (step.cost) {
      text += " · " + formatNumber(step.cost) + " XAF";
    }
    $("vAct").textContent = text;
  }
  if (errorText) {
    $("vAct").textContent = errorText;
  }
  $("vBudget").textContent = money(step.budgetAfter);

  // While a week is moving we show what the farmer saw, and when it is over we show what the farmer sees now.
  let shownObs = step.obs;
  if (atWeekEnd) {
    shownObs = nextObs;
  }
  $("vDet").textContent = Math.round(observationValue(shownObs, "detected_disease") * 100) + "%";
  $("obs").innerHTML = observationRows(shownObs);

  $("vWr").textContent = money(step.reward * 10000);
  $("vWr").className = "val " + (step.reward < 0 ? "neg" : step.reward > 0 ? "pos" : "");
  $("vWrs").textContent = "score " + step.reward.toFixed(2);
  $("vTr").textContent = money(step.totalReturn * 10000);
  $("vTrs").textContent = "score " + step.totalReturn.toFixed(2);
  updateNotesPanel(step, atWeekEnd);
  showCard(isOutro);
}

const stage = $("stage");

// Make the 3D view fill the whole window, and the scene and the camera follow the window size.
function fitToWindow() {
  const windowWidth = innerWidth;
  const windowHeight = Math.max(1, innerHeight);
  const scale = Math.min(windowWidth / 1280, windowHeight / 720);
  const stageWidth = windowWidth / scale;
  const stageHeight = windowHeight / scale;

  stage.style.width = stageWidth + "px";
  stage.style.height = stageHeight + "px";
  stage.style.transform = "scale(" + scale + ")";
  renderer.setPixelRatio(scale * (window.devicePixelRatio || 1));
  renderer.setSize(stageWidth, stageHeight, false);
  camera.aspect = stageWidth / stageHeight;
  camera.updateProjectionMatrix();
  lastHudKey = "";
}
addEventListener("resize", fitToWindow);

// The money the farmer has left now.
function budgetNow() {
  if (steps.length === 0) {
    return startState.budget;
  }
  return steps[steps.length - 1].budgetAfter;
}

const SHORT_NAMES = ["Wait", "Drone scout", "Spray", "Irrigate", "Fertilize", "Remove flagged", "Drone scout + remove", "Weed"];

// Show the right buttons and the tip, depending on who is playing.
function updateControls() {
  lastHudKey = "";
  const choosing = mode === "play" && !seasonDone && startState !== null;

  const box = $("actions");
  box.classList.toggle("on", mode === "play" && !seasonDone);
  let html = "";
  if (config) {
    for (let i = 0; i < config.actions.length; i++) {
      const cost = config.actions[i].cost;
      let costText = "free";
      if (cost) {
        costText = formatNumber(cost) + " XAF";
      }
      const disabled = busy || cost > budgetNow();
      html += '<button data-a="' + i + '" class="' + (i === advice ? "rec" : "") + '"' + (disabled ? " disabled" : "") + ">";
      html += SHORT_NAMES[i] + "<small>" + costText + "</small></button>";
    }
  }
  box.innerHTML = html;
  box.querySelectorAll("button").forEach(function (button) {
    button.onclick = function () {
      playWeek(parseInt(button.dataset.a, 10));
    };
  });

  const adviceBox = $("advice");
  adviceBox.classList.toggle("on", choosing && advice >= 0);
  if (choosing && advice >= 0) {
    adviceBox.innerHTML = "Tip: " + farmerPhrase() + " would choose <b>" + config.actions[advice].name + "</b>";
  }

  const watching = mode === "watch";
  $("next").hidden = !watching;
  $("play").hidden = !watching;
  $("next").disabled = seasonDone || (busy && playing);
  $("polLab").textContent = watching ? "Farmer" : "Tips from";
  $("mWatch").classList.toggle("on", watching);
  $("mPlay").classList.toggle("on", !watching);
  $("play").textContent = autoRun ? "❚❚ PAUSE" : "▶ AUTO RUN";
  $("play").classList.toggle("stop", autoRun);
}

// Ask the server which action the trained farmer would choose now.
async function refreshAdvice() {
  advice = -1;
  if (mode === "play" && !seasonDone) {
    try {
      const data = await callApi("/api/advice");
      advice = data.action;
    } catch (error) {
      showError(error);
    }
  }
  updateControls();
}

// Show a message in the top pill when the server did not answer, and let the user try again.
function showError(error) {
  errorText = "PROBLEM WITH THE LOCAL SERVER · " + error.message;
  busy = false;
  playing = false;
  autoRun = false;
  updateControls();
}

// Start a new season with the farmer, the mode and the season number that are chosen now.
async function startSeason() {
  playing = false;
  autoRun = false;
  busy = true;
  stopFrame = -1;
  steps = [];
  seasonDone = false;
  comparison = null;
  failurePenalty = null;
  frame = 0;
  showCard(false);
  updateControls();

  errorText = "";
  let data;
  try {
    data = await callApi("/api/new", { mode, model, seed, test });
  } catch (error) {
    showError(error);
    return;
  }
  config = data.config;
  seasonLabel = data.label;
  startState = {
    plants: data.start.plants,
    obs: data.start.obs,
    moisture: data.start.moisture,
    budget: data.start.budget,
  };
  nextObs = data.next_obs;
  nextNotes = data.next_notes || [];
  nextChances = data.next_chances || null;
  chanceKind = data.chance_kind || "chance";
  trainedFarmer = !!data.trained;
  busy = false;
  await refreshAdvice();
}

// Play the next week with the real environment, and the user gives the action in the I play mode.
async function fetchWeek(action) {
  let body = {};
  if (mode === "play") {
    body = { action: action };
  }
  const data = await callApi("/api/step", body);
  steps.push(readStep(data.step));
  nextObs = data.next_obs;
  if (!data.done) {
    nextNotes = data.next_notes || [];
    nextChances = data.next_chances || null;
  }
  if (data.done) {
    seasonDone = true;
    comparison = data.comparison || null;
    failurePenalty = data.failure_penalty === undefined ? null : data.failure_penalty;
  }
}

// Get the next week from the server and animate it.
async function playWeek(action) {
  if (busy || seasonDone) {
    return;
  }
  busy = true;
  updateControls();
  const index = steps.length;
  try {
    await fetchWeek(action);
  } catch (error) {
    showError(error);
    return;
  }
  frame = weekStartFrame(index);
  stopFrame = -1;
  if (!seasonDone) {
    stopFrame = weekStartFrame(index + 1) - 1;
  }
  playing = true;
  updateControls();
}

// When a week is over, the animation goes on with the next week or waits for the user.
function weekFinished() {
  playing = false;
  busy = false;
  if (autoRun && !seasonDone) {
    playWeek(null);
    return;
  }
  autoRun = false;
  refreshAdvice();
}

$("play").onclick = function () {
  if (autoRun) {
    autoRun = false;
    playing = false;
    updateControls();
    return;
  }
  if (seasonDone && frame >= lastFrameAvailable()) {
    startSeason().then(function () {
      autoRun = true;
      playWeek(null);
    });
    return;
  }
  autoRun = true;
  if (busy) {
    playing = true;
    updateControls();
  } else {
    playWeek(null);
  }
};

$("next").onclick = function () {
  if (seasonDone || (busy && playing)) {
    return;
  }
  autoRun = false;
  if (busy) {
    playing = true;
    updateControls();
  } else {
    playWeek(null);
  }
};

$("restart").onclick = startSeason;

$("mWatch").onclick = function () {
  mode = "watch";
  startSeason();
};

$("mPlay").onclick = function () {
  mode = "play";
  startSeason();
};

$("pol").onchange = function (event) {
  model = event.target.value;
  startSeason();
};

// Move the camera when a camera button is pressed.
function moveCamera(command) {
  if (command === "left") {
    cameraTurn += 0.06;
  }
  if (command === "right") {
    cameraTurn -= 0.06;
  }
  if (command === "up") {
    cameraHeight = clamp(cameraHeight + 0.04, 0.15, 1.3);
  }
  if (command === "down") {
    cameraHeight = clamp(cameraHeight - 0.04, 0.15, 1.3);
  }
  if (command === "in") {
    cameraDistance = clamp(cameraDistance * 0.95, 10, 40);
  }
  if (command === "out") {
    cameraDistance = clamp(cameraDistance * 1.05, 10, 40);
  }
  if (command === "home") {
    cameraTurn = HOME_TURN;
    cameraHeight = HOME_HEIGHT;
    cameraDistance = HOME_DISTANCE;
  }
}

// Holding a camera button keeps moving the camera.
document.querySelectorAll("#campad button").forEach(function (button) {
  let timer = null;
  function stop() {
    clearInterval(timer);
    timer = null;
  }
  button.addEventListener("pointerdown", function (event) {
    event.preventDefault();
    moveCamera(button.dataset.c);
    if (button.dataset.c !== "home") {
      timer = setInterval(function () {
        moveCamera(button.dataset.c);
      }, 40);
    }
  });
  button.addEventListener("pointerup", stop);
  button.addEventListener("pointerleave", stop);
});

addEventListener("keydown", function (event) {
  const keys = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", "+": "in", "=": "in", "-": "out" };
  if (keys[event.key] && event.target.tagName !== "SELECT") {
    event.preventDefault();
    moveCamera(keys[event.key]);
  }
});

// Dragging the field turns the camera, and the mouse wheel zooms.
let dragStart = null;
$("c").addEventListener("pointerdown", function (event) {
  dragStart = { x: event.clientX, y: event.clientY };
  $("c").setPointerCapture(event.pointerId);
});
$("c").addEventListener("pointermove", function (event) {
  if (!dragStart) {
    return;
  }
  cameraTurn += (event.clientX - dragStart.x) * 0.006;
  cameraHeight = clamp(cameraHeight + (event.clientY - dragStart.y) * 0.004, 0.15, 1.3);
  dragStart = { x: event.clientX, y: event.clientY };
});
$("c").addEventListener("pointerup", function () {
  dragStart = null;
});
$("c").addEventListener(
  "wheel",
  function (event) {
    event.preventDefault();
    cameraDistance = clamp(cameraDistance * (1 + Math.sign(event.deltaY) * 0.08), 10, 40);
  },
  { passive: false }
);

// The animation loop runs about 60 times a second.
let lastTimestamp = performance.now();
function animate(timestamp) {
  requestAnimationFrame(animate);
  const elapsed = Math.min(100, timestamp - lastTimestamp);
  lastTimestamp = timestamp;

  if (playing) {
    frame += (elapsed / 1000) * FPS;
    if (stopFrame >= 0 && frame >= stopFrame) {
      frame = stopFrame;
      weekFinished();
    } else if (frame >= lastFrameAvailable()) {
      frame = lastFrameAvailable();
      playing = false;
      busy = false;
      autoRun = false;
      updateControls();
    }
  }
  drawFrame(Math.floor(frame));
}

// Start from the address of the page, and the at and actions options are only for screenshots.
async function start() {
  const select = $("pol");
  if (!select.querySelector('option[value="' + model + '"]')) {
    const extra = document.createElement("option");
    extra.value = model;
    extra.textContent = model;
    select.appendChild(extra);
  }
  select.value = model;
  fitToWindow();
  requestAnimationFrame(animate);

  await startSeason();

  if (urlParams.get("at") !== null) {
    const weeks = parseFloat(urlParams.get("at"));
    while (!seasonDone && steps.length < Math.ceil(weeks)) {
      await fetchWeek(null);
    }
    frame = Math.min(lastFrameAvailable(), Math.round((INTRO_SECONDS + weeks * SECONDS_PER_WEEK) * FPS));
    updateControls();
  }

  if (urlParams.get("actions") !== null) {
    for (const text of urlParams.get("actions").split(",")) {
      if (seasonDone) {
        break;
      }
      await fetchWeek(parseInt(text, 10));
    }
    frame = lastFrameAvailable();
    await refreshAdvice();
  }
}
start();
