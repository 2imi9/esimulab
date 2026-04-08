/* global Cesium */

// --- CesiumJS Ion token ---
// Get a free token at https://ion.cesium.com/tokens
// For development, the default token provides basic imagery
Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI2ZGIyYTM2Ni1jYTNjLTRiOWQtOTM3NS1mYjhkY2Y5NjgyZTkiLCJpZCI6NDE1NTkyLCJpYXQiOjE3NzU2ODExMDF9.IYSCSnkuW93o-zfmonZaeKEskBQZtCKNGlFm7WvDDp8';

// --- Viewer Setup ---
const viewer = new Cesium.Viewer('cesiumContainer', {
  terrain: Cesium.Terrain.fromWorldTerrain(),
  animation: false,
  timeline: false,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
});

// Set initial view
viewer.camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(-100, 30, 15000000),
  duration: 0,
});

// Enable atmosphere and lighting
viewer.scene.globe.enableLighting = true;

// --- State ---
let drawingMode = false;
let firstCorner = null;
let rectangleEntity = null;
let previewEntity = null;
let selectedBounds = null;

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

// --- Drawing ---
window.startDrawing = function () {
  clearSelection();
  drawingMode = true;
  firstCorner = null;
  viewer.scene.canvas.style.cursor = 'crosshair';

  document.getElementById('btn-draw').style.display = 'none';
  document.getElementById('instructions').querySelector('.step').textContent =
    'Click the first corner of your region on the globe';
};

handler.setInputAction((click) => {
  if (!drawingMode) return;

  const cartesian = viewer.scene.pickPosition(click.position);
  if (!Cesium.defined(cartesian)) {
    // Try ray-cast to globe
    const ray = viewer.camera.getPickRay(click.position);
    const globePos = viewer.scene.globe.pick(ray, viewer.scene);
    if (!Cesium.defined(globePos)) return;
    handleClick(globePos);
  } else {
    handleClick(cartesian);
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

function handleClick(cartesian) {
  const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
  const lon = Cesium.Math.toDegrees(cartographic.longitude);
  const lat = Cesium.Math.toDegrees(cartographic.latitude);

  if (!firstCorner) {
    firstCorner = { lon, lat };
    document.getElementById('instructions').querySelector('.step').textContent =
      `First corner: ${lat.toFixed(3)}, ${lon.toFixed(3)} — Now click the opposite corner`;
  } else {
    const west = Math.min(firstCorner.lon, lon);
    const east = Math.max(firstCorner.lon, lon);
    const south = Math.min(firstCorner.lat, lat);
    const north = Math.max(firstCorner.lat, lat);

    selectedBounds = { west, south, east, north };
    drawRectangle(west, south, east, north);
    finishDrawing();
  }
}

// Live preview rectangle while moving mouse after first click
handler.setInputAction((movement) => {
  if (!drawingMode || !firstCorner) return;

  const cartesian = viewer.scene.pickPosition(movement.endPosition);
  if (!Cesium.defined(cartesian)) return;

  const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
  const lon = Cesium.Math.toDegrees(cartographic.longitude);
  const lat = Cesium.Math.toDegrees(cartographic.latitude);

  const west = Math.min(firstCorner.lon, lon);
  const east = Math.max(firstCorner.lon, lon);
  const south = Math.min(firstCorner.lat, lat);
  const north = Math.max(firstCorner.lat, lat);

  if (previewEntity) {
    viewer.entities.remove(previewEntity);
  }

  previewEntity = viewer.entities.add({
    rectangle: {
      coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
      material: Cesium.Color.CORNFLOWERBLUE.withAlpha(0.2),
      outline: true,
      outlineColor: Cesium.Color.WHITE.withAlpha(0.5),
      outlineWidth: 1,
    },
  });
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

function drawRectangle(west, south, east, north) {
  if (previewEntity) {
    viewer.entities.remove(previewEntity);
    previewEntity = null;
  }

  rectangleEntity = viewer.entities.add({
    rectangle: {
      coordinates: Cesium.Rectangle.fromDegrees(west, south, east, north),
      material: Cesium.Color.CORNFLOWERBLUE.withAlpha(0.3),
      outline: true,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
      height: 0,
    },
  });
}

function finishDrawing() {
  drawingMode = false;
  viewer.scene.canvas.style.cursor = 'default';

  document.getElementById('btn-draw').style.display = 'none';
  document.getElementById('btn-clear').style.display = 'inline-block';
  document.getElementById('btn-confirm').disabled = false;

  updateRegionInfo();

  document.getElementById('instructions').querySelector('.step').textContent =
    'Region selected! Click "Explore Region" to generate 3D terrain';
}

window.clearSelection = function () {
  if (rectangleEntity) {
    viewer.entities.remove(rectangleEntity);
    rectangleEntity = null;
  }
  if (previewEntity) {
    viewer.entities.remove(previewEntity);
    previewEntity = null;
  }

  selectedBounds = null;
  firstCorner = null;
  drawingMode = false;
  viewer.scene.canvas.style.cursor = 'default';

  document.getElementById('btn-draw').style.display = 'inline-block';
  document.getElementById('btn-clear').style.display = 'none';
  document.getElementById('btn-confirm').disabled = true;
  document.getElementById('region-info').style.display = 'none';

  document.getElementById('instructions').querySelector('.step').textContent =
    'Click "Draw Region" then click two corners on the globe to select an area';
};

function updateRegionInfo() {
  if (!selectedBounds) return;

  const { west, south, east, north } = selectedBounds;
  document.getElementById('info-north').textContent = `North: ${north.toFixed(4)}°`;
  document.getElementById('info-south').textContent = `South: ${south.toFixed(4)}°`;
  document.getElementById('info-east').textContent = `East: ${east.toFixed(4)}°`;
  document.getElementById('info-west').textContent = `West: ${west.toFixed(4)}°`;

  // Approximate size in km
  const latKm = (north - south) * 111.32;
  const lonKm = (east - west) * 111.32 * Math.cos(((north + south) / 2) * Math.PI / 180);
  document.getElementById('info-size').textContent = `${lonKm.toFixed(1)} x ${latKm.toFixed(1)} km`;

  document.getElementById('region-info').style.display = 'block';
}

// --- Confirm & Transition ---
window.confirmSelection = async function () {
  if (!selectedBounds) return;

  const { west, south, east, north } = selectedBounds;
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('active');

  document.getElementById('loading-status').textContent = 'Requesting terrain data...';

  try {
    const resp = await fetch('/api/region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ west, south, east, north }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Region fetch failed');
    }

    const data = await resp.json();
    document.getElementById('loading-status').textContent = 'Terrain ready! Redirecting...';

    // Redirect to Three.js viewer with bbox
    await new Promise(r => setTimeout(r, 500));
    window.location.href = `/viewer?bbox=${west},${south},${east},${north}`;

  } catch (err) {
    overlay.classList.remove('active');
    alert(`Error: ${err.message}`);
  }
};
