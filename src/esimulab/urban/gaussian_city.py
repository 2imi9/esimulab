"""GaussianCity integration for photorealistic 3D city generation.

GaussianCity (Xie et al., CVPR 2025) generates photorealistic unbounded
3D city scenes from OSM building footprints using Gaussian Splatting.

Requirements:
- Docker container with CUDA 11.8 (incompatible with Blackwell cu130)
- Pretrained models: rest.pth (BG) + bldg.pth (BLDG) from HuggingFace
- Custom CUDA extensions: diff_gaussian_rasterization, grid_encoder

This module provides:
1. Docker-based inference (isolated CUDA 11.8 environment)
2. OSM data preparation (reuses our Overture Maps data)
3. Output integration (rendered frames → web viewer)

Paper: https://arxiv.org/abs/2406.06526
Code: https://github.com/hzxie/GaussianCity
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GAUSSIANCITY_DOCKER_IMAGE = "esimulab/gaussiancity:latest"
HUGGINGFACE_REPO = "hzxie/GaussianCity"
MODEL_FILES = {
    "bg": "BG-Generator.pth",
    "bldg": "BLDG-Generator.pth",
}


@dataclass
class GaussianCityConfig:
    """Configuration for GaussianCity inference."""

    model_dir: str = "models/gaussiancity"
    output_dir: str = "data/gaussiancity"
    n_frames: int = 24  # number of rendered frames
    sensor_size: tuple[int, int] = (960, 540)
    radius: int = 512  # camera orbit radius
    altitude: int = 640  # camera altitude
    use_docker: bool = True  # always True on Blackwell (needs CUDA 11.8)


def check_availability() -> dict[str, Any]:
    """Check if GaussianCity can run.

    Returns:
        Dict with 'available', 'reason', 'docker', 'models'.
    """
    info: dict[str, Any] = {
        "available": False,
        "reason": "",
        "docker_available": False,
        "models_downloaded": False,
    }

    # Check Docker
    import shutil

    if shutil.which("docker"):
        info["docker_available"] = True
    else:
        info["reason"] = "Docker not available (required for CUDA 11.8 isolation)"
        return info

    # Check models
    model_dir = Path("models/gaussiancity")
    bg_exists = (model_dir / "rest.pth").exists()
    bldg_exists = (model_dir / "bldg.pth").exists()
    info["models_downloaded"] = bg_exists and bldg_exists

    if not info["models_downloaded"]:
        info["reason"] = (
            "Pretrained models not found. Run: "
            "esimulab gaussiancity download-models"
        )
        return info

    info["available"] = True
    info["reason"] = "GaussianCity ready (Docker + models)"
    return info


def download_models(output_dir: str = "models/gaussiancity") -> bool:
    """Download pretrained GaussianCity models from HuggingFace.

    Downloads BG-Generator.pth (~1GB) and BLDG-Generator.pth (~1GB).

    Args:
        output_dir: Directory to save models.

    Returns:
        True if successful.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        for name, filename in MODEL_FILES.items():
            target = out / filename.lower().replace("-", "_").replace(".pth", ".pth")
            if target.exists():
                logger.info("Model %s already exists: %s", name, target)
                continue

            logger.info("Downloading %s from %s...", filename, HUGGINGFACE_REPO)
            downloaded = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=filename,
                local_dir=str(out),
            )
            logger.info("Downloaded: %s", downloaded)

        # Rename to expected names
        for src_name in ["BG-Generator.pth", "BLDG-Generator.pth"]:
            src = out / src_name
            if src.exists():
                dst_name = "rest.pth" if "BG" in src_name else "bldg.pth"
                dst = out / dst_name
                if not dst.exists():
                    src.rename(dst)

        return True

    except ImportError:
        logger.error("huggingface_hub not installed. pip install huggingface_hub")
        return False
    except Exception:
        logger.exception("Model download failed")
        return False


def prepare_osm_projections(
    bbox: tuple[float, float, float, float],
    output_dir: str = "data/gaussiancity/projections",
) -> Path | None:
    """Convert Overture Maps building data to GaussianCity projection format.

    GaussianCity expects:
    - SEG.npy: semantic segmentation map (H, W) uint8 with 8 classes
    - TD_HF.npy: top-down height field (H, W) float32
    - CENTERS.pkl: dict of instance_id → (cx, cy, w, h, d)

    We generate these from our Overture Maps building data.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        output_dir: Output directory for projection files.

    Returns:
        Path to projection directory, or None on failure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from esimulab.urban.overture import fetch_overture_buildings

        dataset = fetch_overture_buildings(bbox, max_buildings=2000)
        if dataset.count == 0:
            logger.warning("No buildings found for bbox %s", bbox)
            return None

        proj_size = 2048
        west, south, east, north = bbox

        # Create semantic segmentation map
        # Classes: 0=NULL, 1=ROAD, 2=BLDG_FACADE, 3=BLDG_ROOF, 4=VEGETATION,
        #          5=WATER, 6=ZONE, 7=SKY
        seg_map = np.full((proj_size, proj_size), 6, dtype=np.uint8)  # default: zone

        # Create height field
        height_field = np.zeros((proj_size, proj_size), dtype=np.float32)

        # Instance centers
        import pickle

        centers = {}

        for idx, b in enumerate(dataset.buildings):
            lon, lat = b.centroid
            # Map to projection coordinates
            px = int((lon - west) / (east - west) * proj_size)
            py = int((lat - south) / (north - south) * proj_size)

            if not (0 <= px < proj_size and 0 <= py < proj_size):
                continue

            # Building footprint (approximate as rectangle)
            half_size = max(3, int(b.height / 5))  # larger buildings = bigger footprint
            x0, x1 = max(0, px - half_size), min(proj_size, px + half_size)
            y0, y1 = max(0, py - half_size), min(proj_size, py + half_size)

            inst_id = 100 + idx * 2  # even = facade, odd = roof
            seg_map[y0:y1, x0:x1] = 2  # BLDG_FACADE
            seg_map[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = 3  # BLDG_ROOF (inner)
            height_field[y0:y1, x0:x1] = b.height

            centers[inst_id] = (px, py, x1 - x0, y1 - y0, b.height)
            centers[inst_id + 1] = (px, py, x1 - x0, y1 - y0, b.height)

        # Add road-like areas between buildings
        road_mask = (seg_map == 6) & (height_field == 0)
        seg_map[road_mask] = 1  # ROAD

        # Save
        proj_dir = out / "Projection"
        proj_dir.mkdir(parents=True, exist_ok=True)
        np.save(proj_dir / "SEG.npy", seg_map)
        np.save(proj_dir / "TD_HF.npy", height_field)

        with open(out / "CENTERS.pkl", "wb") as f:
            pickle.dump(centers, f)

        # Metadata
        metadata = {
            "bbox": list(bbox),
            "building_count": dataset.count,
            "proj_size": proj_size,
        }
        (out / "metadata.json").write_text(json.dumps(metadata))

        logger.info(
            "Prepared GaussianCity projections: %d buildings, %d instances",
            dataset.count,
            len(centers),
        )
        return out

    except Exception:
        logger.exception("OSM projection preparation failed")
        return None


def run_inference_docker(
    projection_dir: str,
    config: GaussianCityConfig | None = None,
) -> list[Path] | None:
    """Run GaussianCity inference in Docker container.

    The Docker container provides CUDA 11.8 + compiled extensions.

    Args:
        projection_dir: Path to prepared projection data.
        config: Inference configuration.

    Returns:
        List of rendered frame paths, or None on failure.
    """
    config = config or GaussianCityConfig()
    import subprocess

    # Check Docker
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("Docker not available")
        return None

    output_dir = Path(config.output_dir) / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{Path(projection_dir).resolve()}:/data/input",
        "-v",
        f"{output_dir.resolve()}:/data/output",
        "-v",
        f"{Path(config.model_dir).resolve()}:/models",
        GAUSSIANCITY_DOCKER_IMAGE,
        "python3",
        "scripts/inference.py",
        "--data_dir",
        "/data/input",
        "--rest_ckpt",
        "/models/rest.pth",
        "--bldg_ckpt",
        "/models/bldg.pth",
        "--output_file",
        "/data/output/rendering.mp4",
    ]

    logger.info("Running GaussianCity inference in Docker...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            logger.error("GaussianCity Docker failed: %s", result.stderr[-500:])
            return None

        # Collect output frames
        frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
        video = output_dir / "rendering.mp4"

        if video.exists():
            logger.info("GaussianCity rendering complete: %s", video)

        if frames:
            logger.info("GaussianCity: %d frames rendered", len(frames))
            return frames

        return [video] if video.exists() else None

    except subprocess.TimeoutExpired:
        logger.error("GaussianCity inference timed out (>10 min)")
        return None
    except Exception:
        logger.exception("GaussianCity inference failed")
        return None


def generate_city_preview(
    bbox: tuple[float, float, float, float],
    config: GaussianCityConfig | None = None,
) -> dict[str, Any]:
    """High-level API: generate photorealistic city from bbox.

    1. Fetch Overture Maps buildings
    2. Prepare GaussianCity projections
    3. Run inference in Docker
    4. Return rendered frames

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        config: Configuration.

    Returns:
        Dict with 'frames', 'video', 'metadata'.
    """
    config = config or GaussianCityConfig()
    result: dict[str, Any] = {"frames": [], "video": None, "metadata": {}}

    # Step 1: Check availability
    status = check_availability()
    if not status["available"]:
        logger.warning("GaussianCity not available: %s", status["reason"])
        result["metadata"]["error"] = status["reason"]
        return result

    # Step 2: Prepare projections
    proj_dir = prepare_osm_projections(bbox, config.output_dir + "/projections")
    if proj_dir is None:
        result["metadata"]["error"] = "Projection preparation failed"
        return result

    # Step 3: Run inference
    frames = run_inference_docker(str(proj_dir), config)
    if frames:
        result["frames"] = [str(f) for f in frames]
        video = Path(config.output_dir) / "renders" / "rendering.mp4"
        if video.exists():
            result["video"] = str(video)

    result["metadata"] = {
        "bbox": list(bbox),
        "n_frames": len(result["frames"]),
        "source": "GaussianCity (Xie et al., CVPR 2025)",
    }

    return result
