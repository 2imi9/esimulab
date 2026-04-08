"""FastAPI server for the Esimulab web viewer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path("data")

app = FastAPI(title="Esimulab Viewer", version="0.1.0")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


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


@app.get("/api/metadata")
async def get_simulation_metadata():
    """Return simulation configuration metadata."""
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return {"status": "no_simulation"}

    return json.loads(meta_path.read_text())


# Mount static files last so API routes take priority
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
