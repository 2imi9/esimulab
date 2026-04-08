"""FastAPI server for the Esimulab web viewer."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path("data")

app = FastAPI(title="Esimulab Viewer", version="0.1.0")


# --- Models ---


class RegionRequest(BaseModel):
    """Bounding box for region selection."""

    west: float
    south: float
    east: float
    north: float


# --- Globe & Viewer Pages ---


@app.get("/", response_class=HTMLResponse)
async def globe_page():
    """Serve the CesiumJS globe landing page."""
    globe_path = STATIC_DIR / "globe.html"
    if not globe_path.exists():
        raise HTTPException(404, "Globe page not found")
    return HTMLResponse(content=globe_path.read_text(encoding="utf-8"))


@app.get("/viewer", response_class=HTMLResponse)
async def viewer_page():
    """Serve the Three.js terrain viewer page."""
    viewer_path = STATIC_DIR / "index.html"
    if not viewer_path.exists():
        raise HTTPException(404, "Viewer page not found")
    return HTMLResponse(content=viewer_path.read_text(encoding="utf-8"))


# --- Health ---


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


# --- Region API ---


@app.post("/api/region")
async def select_region(region: RegionRequest):
    """Fetch terrain for a selected globe region.

    Triggers DEM download and conversion for the given bounding box.
    Returns metadata about the fetched terrain.
    """
    bbox = (region.west, region.south, region.east, region.north)

    # Validate bounds
    if region.west >= region.east or region.south >= region.north:
        raise HTTPException(400, "Invalid bounds: west < east and south < north required")

    if abs(region.east - region.west) > 5 or abs(region.north - region.south) > 5:
        raise HTTPException(400, "Region too large (max 5 degrees per side)")

    try:
        from esimulab.terrain import fetch_dem, prepare_heightfield

        dem_result = fetch_dem(bbox)
        heightfield = prepare_heightfield(dem_result.heightfield, dem_result.pixel_size)

        terrain_dir = DATA_DIR / "terrain"
        terrain_dir.mkdir(parents=True, exist_ok=True)
        np.save(terrain_dir / "heightfield.npy", heightfield.height_field)

        meta = {
            "pixel_size": heightfield.horizontal_scale,
            "vertical_scale": heightfield.vertical_scale,
            "rows": heightfield.height_field.shape[0],
            "cols": heightfield.height_field.shape[1],
            "origin": list(heightfield.origin),
            "bounds_min": list(heightfield.bounds_min),
            "bounds_max": list(heightfield.bounds_max),
            "bbox": list(bbox),
        }
        (terrain_dir / "metadata.json").write_text(json.dumps(meta))

        logger.info("Region terrain fetched: %s, shape=%s", bbox, heightfield.height_field.shape)
        return {"status": "ok", "metadata": meta}

    except Exception as e:
        logger.exception("Region terrain fetch failed")
        raise HTTPException(500, f"Terrain fetch failed: {e}") from e


# --- Terrain API ---


@app.get("/api/terrain")
async def get_terrain():
    """Return terrain heightfield as binary Float32Array."""
    hf_path = DATA_DIR / "terrain" / "heightfield.npy"
    if not hf_path.exists():
        raise HTTPException(404, "No terrain data available")

    hf = np.load(hf_path)
    rows, cols = hf.shape
    header = struct.pack("<II", rows, cols)
    body = hf.astype(np.float32).tobytes()
    return Response(content=header + body, media_type="application/octet-stream")


@app.get("/api/terrain/metadata")
async def get_terrain_metadata():
    """Return terrain metadata as JSON."""
    meta_path = DATA_DIR / "terrain" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "No terrain metadata available")

    return json.loads(meta_path.read_text())


# --- Frame API ---


@app.get("/api/frames/{frame_id}")
async def get_frame(frame_id: str):
    """Return a particle frame as binary data."""
    frame_path = DATA_DIR / "frames" / f"{frame_id}.bin"
    if not frame_path.exists():
        raise HTTPException(404, f"Frame {frame_id} not found")

    return FileResponse(frame_path, media_type="application/octet-stream")


@app.get("/api/frames")
async def list_frames():
    """List available particle frames."""
    frames_dir = DATA_DIR / "frames"
    if not frames_dir.exists():
        return {"frames": []}

    frames = sorted(f.stem for f in frames_dir.glob("frame_*.bin"))
    return {"frames": frames, "count": len(frames)}


# --- Metadata API ---


@app.get("/api/metadata")
async def get_simulation_metadata():
    """Return simulation configuration metadata."""
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return {"status": "no_simulation"}

    return json.loads(meta_path.read_text())


# Mount static files for JS/CSS/assets (but not HTML pages — those are routed above)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static-assets")
# Also mount at root for direct file access (js/globe.js, js/main.js, etc.)
app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
app.mount("/shaders", StaticFiles(directory=str(STATIC_DIR / "shaders")), name="shaders")
