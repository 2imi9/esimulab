import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb);
scene.fog = new THREE.FogExp2(0x87ceeb, 0.0008);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);
camera.position.set(500, -400, 300);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.getElementById('canvas-container').prepend(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 50);
controls.update();

// Lighting
const ambientLight = new THREE.AmbientLight(0x6688cc, 0.5);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
sunLight.position.set(200, -100, 500);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
scene.add(sunLight);

// State
let terrainMesh = null;
let particleMesh = null;
let frames = [];
let currentFrame = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let lastFrameTime = 0;

// Terrain loading
async function loadTerrain() {
  try {
    const resp = await fetch('/api/terrain');
    if (!resp.ok) {
      createPlaceholderTerrain();
      return;
    }

    const buffer = await resp.arrayBuffer();
    const view = new DataView(buffer);
    const rows = view.getUint32(0, true);
    const cols = view.getUint32(4, true);
    const heightData = new Float32Array(buffer, 8);

    const metaResp = await fetch('/api/terrain/metadata');
    const meta = metaResp.ok ? await metaResp.json() : { pixel_size: 30, vertical_scale: 1 };

    const geometry = new THREE.PlaneGeometry(
      cols * meta.pixel_size,
      rows * meta.pixel_size,
      cols - 1,
      rows - 1
    );

    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < heightData.length; i++) {
      vertices[i * 3 + 2] = heightData[i] * meta.vertical_scale;
    }
    geometry.computeVertexNormals();
    geometry.attributes.position.needsUpdate = true;

    // Color by elevation
    const colors = new Float32Array(heightData.length * 3);
    const minH = Math.min(...heightData);
    const maxH = Math.max(...heightData);
    const range = maxH - minH || 1;

    for (let i = 0; i < heightData.length; i++) {
      const t = (heightData[i] - minH) / range;
      // Green-brown gradient
      colors[i * 3] = 0.2 + t * 0.5;
      colors[i * 3 + 1] = 0.5 - t * 0.2;
      colors[i * 3 + 2] = 0.1 + t * 0.1;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.9,
      metalness: 0.0,
      flatShading: false,
    });

    terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.receiveShadow = true;
    scene.add(terrainMesh);
  } catch {
    createPlaceholderTerrain();
  }
}

function createPlaceholderTerrain() {
  const size = 1000;
  const segments = 64;
  const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
  const vertices = geometry.attributes.position.array;

  for (let i = 0; i < vertices.length / 3; i++) {
    const x = vertices[i * 3];
    const y = vertices[i * 3 + 1];
    vertices[i * 3 + 2] = Math.sin(x * 0.01) * Math.cos(y * 0.01) * 30 + 20;
  }
  geometry.computeVertexNormals();
  geometry.attributes.position.needsUpdate = true;

  const material = new THREE.MeshStandardMaterial({
    color: 0x4a7c4f,
    roughness: 0.9,
    flatShading: false,
  });

  terrainMesh = new THREE.Mesh(geometry, material);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);
}

// Particle system
function initParticles(maxCount = 100000) {
  const sphereGeom = new THREE.SphereGeometry(0.5, 4, 4);
  const material = new THREE.MeshStandardMaterial({
    color: 0x4488ff,
    transparent: true,
    opacity: 0.7,
    roughness: 0.3,
    metalness: 0.1,
  });

  particleMesh = new THREE.InstancedMesh(sphereGeom, material, maxCount);
  particleMesh.count = 0;
  scene.add(particleMesh);
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
  document.getElementById('particle-count').textContent = `Particles: ${count}`;
}

// Frame loading
async function loadFrameList() {
  try {
    const resp = await fetch('/api/frames');
    if (!resp.ok) return;
    const data = await resp.json();
    frames = data.frames;
    const slider = document.getElementById('frame-slider');
    slider.max = Math.max(0, frames.length - 1);
    document.getElementById('frame-counter').textContent = `Frame: 0/${frames.length}`;
  } catch {
    // No frames available yet
  }
}

async function loadFrame(index) {
  if (index >= frames.length) return;
  try {
    const resp = await fetch(`/api/frames/${frames[index]}`);
    if (!resp.ok) return;
    const buffer = await resp.arrayBuffer();
    const view = new DataView(buffer);
    const n = view.getUint32(0, true);
    const positions = new Float32Array(buffer, 4, n * 3);
    updateParticles(positions);
    document.getElementById('frame-counter').textContent = `Frame: ${index + 1}/${frames.length}`;
  } catch {
    // Frame load failed silently
  }
}

// Controls
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

// Render loop
let frameCount = 0;
let lastFpsTime = performance.now();

function animate(time) {
  requestAnimationFrame(animate);

  // FPS counter
  frameCount++;
  if (time - lastFpsTime > 1000) {
    document.getElementById('fps-counter').textContent = `FPS: ${frameCount}`;
    frameCount = 0;
    lastFpsTime = time;
  }

  // Playback
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

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Init
async function init() {
  document.getElementById('loading').style.display = 'none';
  initParticles();
  await loadTerrain();
  await loadFrameList();
  animate(0);
}

init();
