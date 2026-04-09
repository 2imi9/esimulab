/* global Cesium */

// --- CesiumJS Ion token ---
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

viewer.camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(-100, 30, 15000000),
  duration: 0,
});
viewer.scene.globe.enableLighting = true;

// --- State ---
let drawingMode = false;
let firstCorner = null;      // {lon, lat}
let currentMouse = null;     // {lon, lat} — tracks mouse in real-time
let rectangleEntity = null;
let firstMarker = null;
let selectedBounds = null;

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

// --- Helper: pick lon/lat from screen position ---
function pickGlobeCoords(screenPos) {
  const ray = viewer.camera.getPickRay(screenPos);
  if (!Cesium.defined(ray)) return null;
  const hit = viewer.scene.globe.pick(ray, viewer.scene);
  if (!Cesium.defined(hit)) return null;
  const carto = Cesium.Cartographic.fromCartesian(hit);
  return {
    lon: Cesium.Math.toDegrees(carto.longitude),
    lat: Cesium.Math.toDegrees(carto.latitude),
  };
}

// --- Drawing ---
window.startDrawing = function () {
  clearSelection();
  drawingMode = true;
  firstCorner = null;
  currentMouse = null;

  // DON'T lock camera — user can rotate globe freely while in draw mode
  // Corners are picked via LEFT_CLICK which CesiumJS doesn't use for rotation
  // (CesiumJS uses LEFT_DRAG for rotation, LEFT_CLICK is separate)

  document.getElementById('btn-draw').style.display = 'none';
  document.getElementById('instructions').querySelector('.step').textContent =
    'Left-click the first corner on the globe (drag to rotate view)';
};

// LEFT_CLICK picks corners (CesiumJS rotation uses LEFT_DRAG, not click)
handler.setInputAction((click) => {
  if (!drawingMode) return;

  const coords = pickGlobeCoords(click.position);
  if (!coords) return;

  if (!firstCorner) {
    // First corner placed
    firstCorner = coords;

    // Add a visible marker at the first corner
    firstMarker = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(coords.lon, coords.lat),
      point: { pixelSize: 10, color: Cesium.Color.YELLOW, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
      label: {
        text: 'Corner 1',
        font: '12px sans-serif',
        pixelOffset: new Cesium.Cartesian2(0, -20),
        fillColor: Cesium.Color.WHITE,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        outlineWidth: 2,
      },
    });

    // Create live-updating rectangle using CallbackProperty
    rectangleEntity = viewer.entities.add({
      rectangle: {
        coordinates: new Cesium.CallbackProperty(() => {
          if (!firstCorner || !currentMouse) return Cesium.Rectangle.fromDegrees(0, 0, 0, 0);
          const w = Math.min(firstCorner.lon, currentMouse.lon);
          const e = Math.max(firstCorner.lon, currentMouse.lon);
          const s = Math.min(firstCorner.lat, currentMouse.lat);
          const n = Math.max(firstCorner.lat, currentMouse.lat);
          return Cesium.Rectangle.fromDegrees(w, s, e, n);
        }, false),
        material: Cesium.Color.CORNFLOWERBLUE.withAlpha(0.25),
        outline: true,
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
      },
    });

    document.getElementById('instructions').querySelector('.step').textContent =
      `Corner 1: ${coords.lat.toFixed(3)}°, ${coords.lon.toFixed(3)}° — click the opposite corner`;
  } else {
    // Second corner → finalize
    const west = Math.min(firstCorner.lon, coords.lon);
    const east = Math.max(firstCorner.lon, coords.lon);
    const south = Math.min(firstCorner.lat, coords.lat);
    const north = Math.max(firstCorner.lat, coords.lat);

    selectedBounds = { west, south, east, north };

    // Replace callback rectangle with static one
    if (rectangleEntity) viewer.entities.remove(rectangleEntity);
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

    finishDrawing();
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

// Track mouse for live rectangle preview
handler.setInputAction((movement) => {
  if (!drawingMode || !firstCorner) return;
  const coords = pickGlobeCoords(movement.endPosition);
  if (coords) currentMouse = coords;
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

function finishDrawing() {
  drawingMode = false;

  // Remove first corner marker
  if (firstMarker) { viewer.entities.remove(firstMarker); firstMarker = null; }

  document.getElementById('btn-draw').style.display = 'none';
  document.getElementById('btn-clear').style.display = 'inline-block';
  document.getElementById('btn-confirm').disabled = false;

  updateRegionInfo();

  document.getElementById('instructions').querySelector('.step').textContent =
    'Region selected! Configure options below, then click "Explore Region"';

  const configPanel = document.getElementById('sim-config');
  if (configPanel) configPanel.style.display = 'block';
}

window.clearSelection = function () {
  if (rectangleEntity) { viewer.entities.remove(rectangleEntity); rectangleEntity = null; }
  if (firstMarker) { viewer.entities.remove(firstMarker); firstMarker = null; }

  selectedBounds = null;
  firstCorner = null;
  currentMouse = null;
  drawingMode = false;

  document.getElementById('btn-draw').style.display = 'inline-block';
  document.getElementById('btn-clear').style.display = 'none';
  document.getElementById('btn-confirm').disabled = true;
  document.getElementById('region-info').style.display = 'none';
  const cfgPanel = document.getElementById('sim-config');
  if (cfgPanel) cfgPanel.style.display = 'none';

  document.getElementById('instructions').querySelector('.step').textContent =
    'Click "Draw Region" then left-click two corners on the globe';
};

function updateRegionInfo() {
  if (!selectedBounds) return;
  const { west, south, east, north } = selectedBounds;
  document.getElementById('info-north').textContent = `North: ${north.toFixed(4)}°`;
  document.getElementById('info-south').textContent = `South: ${south.toFixed(4)}°`;
  document.getElementById('info-east').textContent = `East: ${east.toFixed(4)}°`;
  document.getElementById('info-west').textContent = `West: ${west.toFixed(4)}°`;

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

  const enableUrban = document.getElementById('cfg-urban')?.checked || false;
  const enableMpm = document.getElementById('cfg-mpm')?.checked || false;
  const steps = parseInt(document.getElementById('cfg-steps')?.value || '100');

  try {
    const resp = await fetch('/api/region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ west, south, east, north, enable_urban: enableUrban, enable_mpm: enableMpm, steps }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Region fetch failed');
    }

    document.getElementById('loading-status').textContent = 'Terrain ready! Redirecting...';
    await new Promise(r => setTimeout(r, 500));
    const params = new URLSearchParams({
      bbox: `${west},${south},${east},${north}`,
      urban: enableUrban ? '1' : '0',
      mpm: enableMpm ? '1' : '0',
      steps: String(steps),
    });
    window.location.href = `/viewer?${params.toString()}`;
  } catch (err) {
    overlay.classList.remove('active');
    alert(`Error: ${err.message}`);
  }
};
