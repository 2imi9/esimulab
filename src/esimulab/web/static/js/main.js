import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Sky } from 'three/addons/objects/Sky.js';

// ── Scene ──────────────────────────────────────────────────
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 500000);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.LinearToneMapping;
renderer.toneMappingExposure = 1.0;
document.getElementById('canvas-container').prepend(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 50;
controls.maxDistance = 300000;

// ── Sky ────────────────────────────────────────────────────
const sky = new Sky();
sky.scale.setScalar(450000);
scene.add(sky);

const skyUniforms = sky.material.uniforms;
skyUniforms['turbidity'].value = 2;
skyUniforms['rayleigh'].value = 1.5;
skyUniforms['mieCoefficient'].value = 0.005;
skyUniforms['mieDirectionalG'].value = 0.8;

const sun = new THREE.Vector3();
const phi = THREE.MathUtils.degToRad(90 - 55);  // elevation ~55°
const theta = THREE.MathUtils.degToRad(200);
sun.setFromSphericalCoords(1, phi, theta);
skyUniforms['sunPosition'].value.copy(sun);

// ── Lighting ───────────────────────────────────────────────
const ambientLight = new THREE.HemisphereLight(0x87ceeb, 0x4a3728, 0.4);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xfff4e0, 2.0);
sunLight.position.copy(sun).multiplyScalar(10000);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(4096, 4096);
sunLight.shadow.camera.near = 1;
sunLight.shadow.camera.far = 50000;
scene.add(sunLight);

// ── State ──────────────────────────────────────────────────
let terrainMesh = null;
let particleMesh = null;
let buildingMesh = null;
let contourLines = null;
let frames = [];
let currentFrame = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let lastFrameTime = 0;
let terrainMeta = null;
let verticalExaggeration = 3.0;
let terrainHeightData = null;
let terrainMinH = 0, terrainMaxH = 100;

// ── Hypsometric color ramp ─────────────────────────────────
function hypsometricColor(elevation, min, max) {
  const t = Math.max(0, Math.min(1, (elevation - min) / (max - min || 1)));
  // High-contrast: vivid green → yellow → orange → brown → grey → white
  const stops = [
    { t: 0.0,  r: 0.08, g: 0.45, b: 0.08 },  // rich green (valleys)
    { t: 0.12, r: 0.20, g: 0.58, b: 0.12 },  // bright green
    { t: 0.25, r: 0.55, g: 0.68, b: 0.15 },  // yellow-green
    { t: 0.40, r: 0.78, g: 0.65, b: 0.20 },  // golden yellow
    { t: 0.55, r: 0.72, g: 0.45, b: 0.18 },  // warm brown
    { t: 0.70, r: 0.58, g: 0.38, b: 0.28 },  // dark brown
    { t: 0.82, r: 0.62, g: 0.58, b: 0.55 },  // grey-brown (rock)
    { t: 0.92, r: 0.82, g: 0.80, b: 0.78 },  // light grey
    { t: 1.0,  r: 0.97, g: 0.97, b: 1.00 },  // snow white
  ];

  let lower = stops[0], upper = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }

  const localT = (t - lower.t) / (upper.t - lower.t || 1);
  return {
    r: lower.r + (upper.r - lower.r) * localT,
    g: lower.g + (upper.g - lower.g) * localT,
    b: lower.b + (upper.b - lower.b) * localT,
  };
}

// ── Terrain ────────────────────────────────────────────────
async function loadTerrain() {
  try {
    const resp = await fetch('/api/terrain');
    if (!resp.ok) { createPlaceholderTerrain(); return; }

    const buffer = await resp.arrayBuffer();
    const view = new DataView(buffer);
    const rows = view.getUint32(0, true);
    const cols = view.getUint32(4, true);
    const heightData = new Float32Array(buffer, 8);

    const metaResp = await fetch('/api/terrain/metadata');
    terrainMeta = metaResp.ok ? await metaResp.json() : { pixel_size: 30, vertical_scale: 1 };
    const ps = terrainMeta.pixel_size;
    const vs = terrainMeta.vertical_scale;

    // Downsample if too large for GPU
    let dRows = rows, dCols = cols, dHeight = heightData;
    const maxDim = 512;
    if (rows > maxDim || cols > maxDim) {
      const stepR = Math.ceil(rows / maxDim);
      const stepC = Math.ceil(cols / maxDim);
      dRows = Math.floor(rows / stepR);
      dCols = Math.floor(cols / stepC);
      dHeight = new Float32Array(dRows * dCols);
      for (let r = 0; r < dRows; r++) {
        for (let c = 0; c < dCols; c++) {
          dHeight[r * dCols + c] = heightData[r * stepR * cols + c * stepC];
        }
      }
      terrainMeta._downsample = { stepR, stepC };
      terrainMeta.pixel_size = ps * Math.max(stepR, stepC);
    }

    const dPs = terrainMeta.pixel_size;
    const geometry = new THREE.PlaneGeometry(
      dCols * dPs, dRows * dPs, dCols - 1, dRows - 1
    );

    // Store raw heights for contour/rebuild
    terrainHeightData = dHeight;

    const vertices = geometry.attributes.position.array;
    let minH = Infinity, maxH = -Infinity;
    for (let i = 0; i < dHeight.length; i++) {
      const h = dHeight[i] * vs;
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;
    }
    terrainMinH = minH;
    terrainMaxH = maxH;

    // Apply vertical exaggeration
    for (let i = 0; i < dHeight.length; i++) {
      const h = dHeight[i] * vs;
      vertices[i * 3 + 2] = h * verticalExaggeration;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    // Update elevation legend
    document.getElementById('elev-max').textContent = `${maxH.toFixed(0)}m`;
    document.getElementById('elev-min').textContent = `${minH.toFixed(0)}m`;

    // Hypsometric vertex coloring
    const colors = new Float32Array(dHeight.length * 3);
    for (let i = 0; i < dHeight.length; i++) {
      const h = dHeight[i] * vs;
      const c = hypsometricColor(h, minH, maxH);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.0,
      flatShading: false,
      side: THREE.DoubleSide,
    });

    terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.receiveShadow = true;
    terrainMesh.castShadow = true;
    scene.add(terrainMesh);

    // Auto-fit camera (use exaggerated heights)
    const extentX = dCols * dPs;
    const extentY = dRows * dPs;
    const diag = Math.sqrt(extentX * extentX + extentY * extentY);
    const exagMax = maxH * verticalExaggeration;
    const exagMid = (minH + maxH) / 2 * verticalExaggeration;
    camera.position.set(extentX * 0.5, -extentY * 0.4, exagMax + diag * 0.2);
    controls.target.set(0, 0, exagMid);
    camera.far = diag * 3;
    camera.updateProjectionMatrix();
    controls.update();

    // Adjust shadow camera to terrain
    const halfDiag = diag / 2;
    sunLight.shadow.camera.left = -halfDiag;
    sunLight.shadow.camera.right = halfDiag;
    sunLight.shadow.camera.top = halfDiag;
    sunLight.shadow.camera.bottom = -halfDiag;
    sunLight.shadow.camera.far = diag * 2;
    sunLight.shadow.camera.updateProjectionMatrix();

    // Fog tuned to terrain
    scene.fog = new THREE.FogExp2(0x9db8d2, 1.5 / diag);

    document.getElementById('loading').textContent =
      `Terrain: ${rows}x${cols} → ${dRows}x${dCols}, elev ${minH.toFixed(0)}–${maxH.toFixed(0)}m`;
    setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 3000);

  } catch (e) {
    console.warn('Terrain load failed, using placeholder:', e);
    createPlaceholderTerrain();
  }
}

function createPlaceholderTerrain() {
  const size = 2000;
  const segments = 128;
  const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
  const vertices = geometry.attributes.position.array;
  const colors = new Float32Array((segments + 1) * (segments + 1) * 3);

  for (let i = 0; i < vertices.length / 3; i++) {
    const x = vertices[i * 3], y = vertices[i * 3 + 1];
    const h = Math.sin(x * 0.005) * Math.cos(y * 0.005) * 80
            + Math.sin(x * 0.02) * 15 + Math.cos(y * 0.015) * 20 + 100;
    vertices[i * 3 + 2] = h;
    const c = hypsometricColor(h, 0, 200);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  geometry.attributes.position.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  terrainMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.85, flatShading: false,
  }));
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

  camera.position.set(1200, -900, 600);
  controls.target.set(0, 0, 100);
  scene.fog = new THREE.FogExp2(0x9db8d2, 0.0004);
  document.getElementById('loading').style.display = 'none';
}

// ── Water particles ────────────────────────────────────────
function initParticles(maxCount = 100000) {
  const sphereGeom = new THREE.SphereGeometry(1.5, 6, 6);
  const material = new THREE.MeshPhysicalMaterial({
    color: 0x2288dd,
    transparent: true,
    opacity: 0.6,
    roughness: 0.1,
    metalness: 0.0,
    transmission: 0.3,
    thickness: 0.5,
  });

  particleMesh = new THREE.InstancedMesh(sphereGeom, material, maxCount);
  particleMesh.count = 0;
  particleMesh.castShadow = true;
  scene.add(particleMesh);
}

// ── Wind arrow indicator ───────────────────────────────────
let windArrow = null;

function createWindArrow(dir, magnitude) {
  if (windArrow) scene.remove(windArrow);
  if (!terrainMesh || magnitude < 0.1) return;

  const arrowDir = new THREE.Vector3(dir[0], dir[1], 0).normalize();
  const origin = new THREE.Vector3(0, 0, 300);
  const length = Math.min(magnitude * 50, 500);
  windArrow = new THREE.ArrowHelper(arrowDir, origin, length, 0xffaa00, length * 0.2, length * 0.1);
  scene.add(windArrow);
}

function updateParticles(positions) {
  if (!particleMesh) return;
  const count = positions.length / 3;
  particleMesh.count = count;

  const matrix = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    matrix.setPosition(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    particleMesh.setMatrixAt(i, matrix);
  }
  particleMesh.instanceMatrix.needsUpdate = true;
  document.getElementById('particle-count').textContent = `Particles: ${count.toLocaleString()}`;
}

// ── URL bbox parameter ─────────────────────────────────────
function getBboxFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const bbox = params.get('bbox');
  if (!bbox) return null;
  const parts = bbox.split(',').map(Number);
  if (parts.length !== 4 || parts.some(isNaN)) return null;
  return { west: parts[0], south: parts[1], east: parts[2], north: parts[3] };
}

async function fetchRegionTerrain(bbox) {
  document.getElementById('loading').textContent = 'Fetching terrain for selected region...';
  document.getElementById('loading').style.display = 'block';
  try {
    const resp = await fetch('/api/region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bbox),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      document.getElementById('loading').textContent = `Error: ${err.detail}`;
      return false;
    }
    return true;
  } catch (e) {
    document.getElementById('loading').textContent = `Network error: ${e.message}`;
    return false;
  }
}

// ── Weather info panel ─────────────────────────────────────
async function loadWeatherInfo() {
  const infoPanel = document.getElementById('weather-info');
  if (!infoPanel) return;

  try {
    const metaResp = await fetch('/api/metadata');
    const meta = metaResp.ok ? await metaResp.json() : null;

    const terrainResp = await fetch('/api/terrain/metadata');
    const terrainMeta = terrainResp.ok ? await terrainResp.json() : null;

    let html = '';

    if (terrainMeta) {
      const rows = terrainMeta.rows || '?';
      const cols = terrainMeta.cols || '?';
      const ps = terrainMeta.pixel_size?.toFixed(1) || '?';
      html += `<div class="info-label">Terrain</div>`;
      html += `<div class="info-value">${cols}x${rows} @ ${ps}m/px</div>`;

      if (terrainMeta.bounds_min && terrainMeta.bounds_max) {
        const zMin = terrainMeta.bounds_min[2]?.toFixed(0) || '0';
        const zMax = terrainMeta.bounds_max[2]?.toFixed(0) || '0';
        html += `<div class="info-value">Elev: ${zMin}–${zMax} m</div>`;
      }
    }

    if (meta && meta.bbox) {
      const [w, s, e, n] = meta.bbox;
      html += `<div class="info-label" style="margin-top:6px">Region</div>`;
      html += `<div class="info-value">${n.toFixed(3)}°N ${w.toFixed(3)}°W</div>`;
    }

    // Try wind data from simulation metadata
    try {
      const simResp = await fetch('/api/metadata');
      if (simResp.ok) {
        const simMeta = await simResp.json();
        if (simMeta.wind_magnitude) {
          html += `<div class="info-label" style="margin-top:6px">Wind</div>`;
          html += `<div class="info-value">${simMeta.wind_magnitude.toFixed(1)} m/s</div>`;
        }
        if (simMeta.temperature_k) {
          const tC = (simMeta.temperature_k - 273.15).toFixed(1);
          html += `<div class="info-label" style="margin-top:6px">Temperature</div>`;
          html += `<div class="info-value">${tC}°C</div>`;
        }
        if (simMeta.precipitation_mm_hr !== undefined) {
          html += `<div class="info-label" style="margin-top:6px">Precipitation</div>`;
          html += `<div class="info-value">${simMeta.precipitation_mm_hr.toFixed(1)} mm/hr</div>`;
        }
      }
    } catch { /* no wind data */ }

    // Urban data
    try {
      const urbanResp = await fetch('/api/urban/metadata');
      if (urbanResp.ok) {
        const urban = await urbanResp.json();
        if (urban.building_count) {
          html += `<div class="info-label" style="margin-top:6px">Urban</div>`;
          html += `<div class="info-value">${urban.building_count} buildings</div>`;
          if (urban.mean_imperviousness !== undefined) {
            html += `<div class="info-value">Impervious: ${(urban.mean_imperviousness * 100).toFixed(0)}%</div>`;
          }
          if (urban.mean_runoff_coefficient !== undefined) {
            html += `<div class="info-value">Runoff coeff: ${urban.mean_runoff_coefficient.toFixed(2)}</div>`;
          }
        }
      }
    } catch { /* no urban data */ }

    // Overlay links
    try {
      const overlayResp = await fetch('/api/overlays');
      if (overlayResp.ok) {
        const overlays = await overlayResp.json();
        if (overlays.overlays && overlays.overlays.length > 0) {
          html += `<div class="info-label" style="margin-top:6px">Overlays</div>`;
          for (const name of overlays.overlays) {
            html += `<div class="info-value">${name}</div>`;
          }
        }
      }
    } catch { /* no overlays */ }

    if (html) {
      infoPanel.innerHTML = html;
      infoPanel.style.display = 'block';
    }
  } catch { /* ignore */ }
}

// ── Frame loading ──────────────────────────────────────────
async function loadFrameList() {
  try {
    const resp = await fetch('/api/frames');
    if (!resp.ok) return;
    const data = await resp.json();
    frames = data.frames;
    const slider = document.getElementById('frame-slider');
    slider.max = Math.max(0, frames.length - 1);
    document.getElementById('frame-counter').textContent = `Frame: 0/${frames.length}`;
  } catch { /* No frames available yet */ }
}

async function loadFrame(index) {
  if (index >= frames.length) return;
  try {
    // Use subsampled endpoint for web performance (50k particles max)
    const resp = await fetch(`/api/frames/${frames[index]}/subsampled?max_particles=50000`);
    if (!resp.ok) return;
    const buffer = await resp.arrayBuffer();
    const view = new DataView(buffer);
    const n = view.getUint32(0, true);
    const positions = new Float32Array(buffer, 4, n * 3);
    updateParticles(positions);
    document.getElementById('frame-counter').textContent = `Frame: ${index + 1}/${frames.length}`;
  } catch { /* Frame load failed */ }
}

// ── Controls ───────────────────────────────────────────────
document.getElementById('btn-play').addEventListener('click', () => {
  isPlaying = true;
  document.getElementById('btn-play').classList.add('active');
  document.getElementById('btn-pause').classList.remove('active');
});

document.getElementById('btn-pause').addEventListener('click', () => {
  isPlaying = false;
  document.getElementById('btn-pause').classList.add('active');
  document.getElementById('btn-play').classList.remove('active');
});

document.getElementById('speed-slider').addEventListener('input', (e) => {
  playbackSpeed = parseFloat(e.target.value);
});

document.getElementById('frame-slider').addEventListener('input', (e) => {
  currentFrame = parseInt(e.target.value);
  loadFrame(currentFrame);
});

// ── Render loop ────────────────────────────────────────────
let fCount = 0;
let lastFpsTime = performance.now();

function animate(time) {
  requestAnimationFrame(animate);

  fCount++;
  if (time - lastFpsTime > 1000) {
    document.getElementById('fps-counter').textContent = `FPS: ${fCount}`;
    fCount = 0;
    lastFpsTime = time;
  }

  if (isPlaying && frames.length > 0) {
    const interval = 1000 / (30 * playbackSpeed);
    if (time - lastFrameTime > interval) {
      currentFrame = (currentFrame + 1) % frames.length;
      document.getElementById('frame-slider').value = currentFrame;
      loadFrame(currentFrame);
      lastFrameTime = time;
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

// ── Resize ─────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Buildings ──────────────────────────────────────────────
async function loadBuildings() {
  try {
    const resp = await fetch('/api/urban/buildings');
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.buildings || data.buildings.length === 0) return;

    // Calculate terrain center for coordinate mapping
    let originLon = 0, originLat = 0;
    if (terrainMeta && terrainMeta.bbox) {
      const [w, s, e, n] = terrainMeta.bbox;
      originLon = (w + e) / 2;
      originLat = (s + n) / 2;
    }

    const boxGeom = new THREE.BoxGeometry(1, 1, 1);
    const buildingMat = new THREE.MeshStandardMaterial({
      color: 0xd4a574,       // warm sandstone — distinct from green/brown terrain
      roughness: 0.5,
      metalness: 0.1,
      emissive: 0x332211,    // slight warm glow so buildings pop even in shadow
      emissiveIntensity: 0.15,
    });

    const maxBuildings = Math.min(data.buildings.length, 2000);
    buildingMesh = new THREE.InstancedMesh(boxGeom, buildingMat, maxBuildings);
    buildingMesh.castShadow = true;
    buildingMesh.receiveShadow = true;

    const matrix = new THREE.Matrix4();
    let count = 0;

    for (let i = 0; i < maxBuildings; i++) {
      const b = data.buildings[i];
      // Convert lon/lat to local meters
      const dx = (b.lon - originLon) * 111320 * Math.cos(originLat * Math.PI / 180);
      const dy = (b.lat - originLat) * 110540;
      const h = Math.max(b.height, 5) * verticalExaggeration;
      const footprint = Math.max(12, b.height * 0.8); // larger buildings = wider

      // Compose: scale then translate
      matrix.identity();
      matrix.makeScale(footprint, footprint, h);
      matrix.setPosition(dx, dy, h / 2);

      buildingMesh.setMatrixAt(i, matrix);
      count++;
    }

    buildingMesh.count = count;
    buildingMesh.instanceMatrix.needsUpdate = true;
    scene.add(buildingMesh);

    console.log(`Loaded ${count} buildings`);
  } catch (e) {
    console.warn('Building load failed:', e);
  }
}

// ── Contour Lines ──────────────────────────────────────────
function generateContourLines() {
  if (!terrainMesh || !terrainHeightData || !terrainMeta) return;

  // Remove old contours
  if (contourLines) {
    scene.remove(contourLines);
    contourLines = null;
  }

  const dPs = terrainMeta.pixel_size;
  const geom = terrainMesh.geometry;
  const positions = geom.attributes.position.array;
  const cols = Math.round(Math.sqrt(terrainHeightData.length * (terrainMeta.cols / terrainMeta.rows))) || 100;
  const rows = Math.round(terrainHeightData.length / cols) || 100;

  // Contour interval: auto-scale to ~10 contour lines
  const range = terrainMaxH - terrainMinH;
  const interval = Math.max(10, Math.round(range / 10 / 10) * 10); // round to 10m

  const linePoints = [];
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.3, transparent: true });

  // March through grid, find contour crossings
  for (let elev = terrainMinH + interval; elev < terrainMaxH; elev += interval) {
    for (let i = 0; i < positions.length / 3 - 1; i++) {
      const z1 = positions[i * 3 + 2] / verticalExaggeration;
      const z2 = positions[(i + 1) * 3 + 2] / verticalExaggeration;
      if ((z1 - elev) * (z2 - elev) < 0) {
        const t = (elev - z1) / (z2 - z1);
        const x = positions[i * 3] + t * (positions[(i + 1) * 3] - positions[i * 3]);
        const y = positions[i * 3 + 1] + t * (positions[(i + 1) * 3 + 1] - positions[i * 3 + 1]);
        linePoints.push(new THREE.Vector3(x, y, elev * verticalExaggeration + 0.5));
      }
    }
  }

  if (linePoints.length > 1) {
    const pointsGeom = new THREE.BufferGeometry().setFromPoints(linePoints);
    contourLines = new THREE.Points(pointsGeom, new THREE.PointsMaterial({
      color: 0x000000, size: 1.5, opacity: 0.4, transparent: true,
    }));
    scene.add(contourLines);
  }
}

// ── Layer Controls ─────────────────────────────────────────
document.getElementById('slider-exag')?.addEventListener('input', (e) => {
  verticalExaggeration = parseFloat(e.target.value);
  document.getElementById('exag-val').textContent = `${verticalExaggeration}x`;

  // Rebuild terrain Z values
  if (terrainMesh && terrainHeightData && terrainMeta) {
    const verts = terrainMesh.geometry.attributes.position.array;
    const vs = terrainMeta.vertical_scale || 1;
    for (let i = 0; i < terrainHeightData.length; i++) {
      verts[i * 3 + 2] = terrainHeightData[i] * vs * verticalExaggeration;
    }
    terrainMesh.geometry.attributes.position.needsUpdate = true;
    terrainMesh.geometry.computeVertexNormals();
  }

  // Update contours
  if (document.getElementById('chk-contours')?.checked) {
    generateContourLines();
  }
});

document.getElementById('chk-buildings')?.addEventListener('change', (e) => {
  if (buildingMesh) buildingMesh.visible = e.target.checked;
});

document.getElementById('chk-contours')?.addEventListener('change', (e) => {
  if (e.target.checked) {
    generateContourLines();
  } else if (contourLines) {
    scene.remove(contourLines);
    contourLines = null;
  }
});

// ── Init ───────────────────────────────────────────────────
async function init() {
  // Check if bbox was passed from globe selection
  const bbox = getBboxFromUrl();
  if (bbox) {
    const ok = await fetchRegionTerrain(bbox);
    if (!ok) {
      console.warn('Region fetch failed, loading existing terrain');
    }
  }

  initParticles();
  await loadTerrain();
  await loadBuildings();
  await loadFrameList();
  loadWeatherInfo();
  animate(0);
}

init();
