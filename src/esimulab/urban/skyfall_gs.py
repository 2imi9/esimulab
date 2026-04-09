"""Skyfall-GS integration for photorealistic urban representation.

Skyfall-GS (Lee et al., 2025) synthesizes immersive 3D urban scenes from
satellite imagery using Gaussian Splatting with curriculum-driven iterative
refinement. It produces geometrically accurate, photorealistic city scenes.

This is the DEFAULT urban representation layer in Esimulab.

Pipeline:
  1. Download pretrained fused PLY from HuggingFace
  2. Load as 3D Gaussian Splat in viewer (separate urban layer)
  3. Align with terrain coordinate system
  4. Optionally train on custom satellite imagery

Paper: https://arxiv.org/abs/2510.15869
Code: https://github.com/jayin92/Skyfall-GS
License: Apache 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HUGGINGFACE_REPO = "jayin92/Skyfall-GS"
HUGGINGFACE_PLY_REPO = "jayin92/Skyfall-GS"  # PLY models repo

# Available pretrained scenes
PRETRAINED_SCENES = {
    "JAX_068": {
        "city": "Jacksonville",
        "area_km2": 0.25,
        "description": "Urban area with mixed buildings",
    },
    "JAX_004": {"city": "Jacksonville", "area_km2": 0.25, "description": "Residential area"},
    "NYC_004": {"city": "New York City", "area_km2": 0.25, "description": "Manhattan urban"},
    "NYC_010": {"city": "New York City", "area_km2": 0.25, "description": "NYC urban area"},
}


@dataclass
class SkyfallConfig:
    """Configuration for Skyfall-GS urban layer."""

    model_dir: str = "models/skyfall-gs"
    output_dir: str = "data/urban/skyfall"
    # Rendering
    sh_degree: int = 1
    render_resolution: tuple[int, int] = (1024, 1024)
    # Training (for custom scenes)
    iterations_stage1: int = 30000
    iterations_stage2: int = 80000
    # Docker
    use_docker: bool = True
    docker_image: str = "esimulab/skyfall-gs:latest"


@dataclass
class UrbanScene:
    """A loaded Skyfall-GS urban scene."""

    ply_path: Path | None = None
    scene_id: str = ""
    metadata: dict = field(default_factory=dict)
    n_gaussians: int = 0
    bounds_min: tuple[float, float, float] = (0, 0, 0)
    bounds_max: tuple[float, float, float] = (0, 0, 0)
    # Rendered frames for web viewer
    rendered_frames: list[Path] = field(default_factory=list)


def check_availability() -> dict[str, Any]:
    """Check if Skyfall-GS can run."""
    info: dict[str, Any] = {
        "available": False,
        "pretrained_scenes": list(PRETRAINED_SCENES.keys()),
        "downloaded_scenes": [],
        "reason": "",
    }

    model_dir = Path("models/skyfall-gs")
    if model_dir.exists():
        plys = list(model_dir.glob("*_fused.ply"))
        info["downloaded_scenes"] = [p.stem.replace("_fused", "") for p in plys]

    if info["downloaded_scenes"]:
        info["available"] = True
        info["reason"] = f"{len(info['downloaded_scenes'])} pretrained scenes available"
    else:
        info["reason"] = (
            "No pretrained scenes found. Run: esimulab skyfall download --scene JAX_068"
        )

    return info


def download_pretrained_ply(
    scene_id: str = "JAX_068",
    output_dir: str = "models/skyfall-gs",
) -> Path | None:
    """Download a pretrained fused PLY from HuggingFace.

    Args:
        scene_id: Scene identifier (e.g., 'JAX_068', 'NYC_004').
        output_dir: Directory to save the PLY file.

    Returns:
        Path to downloaded PLY, or None on failure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = out / f"{scene_id}_fused.ply"
    if target.exists():
        logger.info("Pretrained PLY already exists: %s", target)
        return target

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s fused PLY from HuggingFace...", scene_id)
        downloaded = hf_hub_download(
            repo_id=HUGGINGFACE_PLY_REPO,
            filename=f"{scene_id}_fused.ply",
            local_dir=str(out),
        )
        logger.info("Downloaded: %s", downloaded)
        return Path(downloaded)

    except ImportError:
        logger.error("huggingface_hub not installed. pip install huggingface_hub")
        return None
    except Exception:
        logger.exception("Download failed for %s", scene_id)
        return None


def load_ply_metadata(ply_path: Path) -> dict[str, Any]:
    """Read metadata from a Gaussian splat PLY file header.

    Args:
        ply_path: Path to fused PLY file.

    Returns:
        Dict with vertex count, properties, bounds.
    """
    meta: dict[str, Any] = {"path": str(ply_path), "n_gaussians": 0, "properties": []}

    try:
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
                if line.startswith(b"element vertex"):
                    meta["n_gaussians"] = int(line.split()[-1])
                if line.startswith(b"property"):
                    meta["properties"].append(line.decode().strip())

        meta["file_size_mb"] = ply_path.stat().st_size / (1024 * 1024)
        logger.info(
            "PLY metadata: %d Gaussians, %.1f MB, %d properties",
            meta["n_gaussians"],
            meta["file_size_mb"],
            len(meta["properties"]),
        )
    except Exception:
        logger.exception("Failed to read PLY metadata")

    return meta


def ply_to_web_splat(
    ply_path: Path,
    output_path: Path | None = None,
    max_gaussians: int = 500000,
) -> Path | None:
    """Convert a Gaussian PLY to a web-viewable .splat format.

    The .splat format is a compact binary used by web Gaussian splat
    viewers (like antimatter15/splat or mkkellogg/GaussianSplats3D).

    Each Gaussian: position(3f) + scale(3f) + color(4B) + rotation(4B) = 32 bytes

    Args:
        ply_path: Input fused PLY file.
        output_path: Output .splat file. Default: same name with .splat extension.
        max_gaussians: Maximum Gaussians to include (subsample if exceeded).

    Returns:
        Path to .splat file, or None on failure.
    """
    if output_path is None:
        output_path = ply_path.with_suffix(".splat")

    try:
        import struct

        # Read PLY
        with open(ply_path, "rb") as f:
            # Skip header
            n_verts = 0
            properties = []
            while True:
                line = f.readline().decode().strip()
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("property"):
                    properties.append(line.split()[-1])
                elif line == "end_header":
                    break

            # Read vertex data
            prop_count = len(properties)
            # Each property is float32 (4 bytes)
            vertex_size = prop_count * 4
            raw = f.read(n_verts * vertex_size)

        data = np.frombuffer(raw, dtype=np.float32).reshape(n_verts, prop_count)

        # Subsample if needed
        if n_verts > max_gaussians:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_verts, max_gaussians, replace=False)
            idx.sort()
            data = data[idx]
            n_verts = max_gaussians

        # Find property indices (standard 3DGS PLY format)
        prop_idx = {name: i for i, name in enumerate(properties)}

        # Extract fields
        x = data[:, prop_idx.get("x", 0)]
        y = data[:, prop_idx.get("y", 1)]
        z = data[:, prop_idx.get("z", 2)]

        # Scale (log scale in PLY)
        sx = (
            np.exp(data[:, prop_idx.get("scale_0", 6)])
            if "scale_0" in prop_idx
            else np.ones(n_verts) * 0.01
        )
        sy = np.exp(data[:, prop_idx.get("scale_1", 7)]) if "scale_1" in prop_idx else sx
        sz = np.exp(data[:, prop_idx.get("scale_2", 8)]) if "scale_2" in prop_idx else sx

        # Color (SH band 0 → RGB)
        sh0_r = data[:, prop_idx.get("f_dc_0", 3)] if "f_dc_0" in prop_idx else np.zeros(n_verts)
        sh0_g = data[:, prop_idx.get("f_dc_1", 4)] if "f_dc_1" in prop_idx else np.zeros(n_verts)
        sh0_b = data[:, prop_idx.get("f_dc_2", 5)] if "f_dc_2" in prop_idx else np.zeros(n_verts)

        # SH to RGB: color = 0.5 + sh_c0 * sh0 where sh_c0 = 0.28209479
        sh_c0 = 0.28209479
        r = np.clip((0.5 + sh_c0 * sh0_r) * 255, 0, 255).astype(np.uint8)
        g = np.clip((0.5 + sh_c0 * sh0_g) * 255, 0, 255).astype(np.uint8)
        b = np.clip((0.5 + sh_c0 * sh0_b) * 255, 0, 255).astype(np.uint8)

        # Opacity (sigmoid in PLY)
        opacity_raw = (
            data[:, prop_idx.get("opacity", 9)] if "opacity" in prop_idx else np.zeros(n_verts)
        )
        opacity = (1 / (1 + np.exp(-opacity_raw)) * 255).astype(np.uint8)

        # Rotation quaternion
        rot0 = data[:, prop_idx.get("rot_0", 10)] if "rot_0" in prop_idx else np.ones(n_verts)
        rot1 = data[:, prop_idx.get("rot_1", 11)] if "rot_1" in prop_idx else np.zeros(n_verts)
        rot2 = data[:, prop_idx.get("rot_2", 12)] if "rot_2" in prop_idx else np.zeros(n_verts)
        rot3 = data[:, prop_idx.get("rot_3", 13)] if "rot_3" in prop_idx else np.zeros(n_verts)
        # Normalize
        norm = np.sqrt(rot0**2 + rot1**2 + rot2**2 + rot3**2) + 1e-8
        rot0, rot1, rot2, rot3 = rot0 / norm, rot1 / norm, rot2 / norm, rot3 / norm
        # Map to 0-255
        rot0_b = ((rot0 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        rot1_b = ((rot1 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        rot2_b = ((rot2 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        rot3_b = ((rot3 * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        # Write .splat format: 32 bytes per Gaussian
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for i in range(n_verts):
                f.write(struct.pack("<fff", x[i], y[i], z[i]))  # 12 bytes position
                f.write(struct.pack("<fff", sx[i], sy[i], sz[i]))  # 12 bytes scale
                f.write(struct.pack("BBBB", r[i], g[i], b[i], opacity[i]))  # 4 bytes color+alpha
                f.write(
                    struct.pack("BBBB", rot0_b[i], rot1_b[i], rot2_b[i], rot3_b[i])
                )  # 4 bytes rotation

        logger.info(
            "Converted to .splat: %d Gaussians, %.1f MB",
            n_verts,
            output_path.stat().st_size / (1024 * 1024),
        )
        return output_path

    except Exception:
        logger.exception("PLY to splat conversion failed")
        return None


def prepare_urban_layer(
    scene_id: str = "JAX_068",
    config: SkyfallConfig | None = None,
) -> UrbanScene | None:
    """Prepare a Skyfall-GS urban layer for the web viewer.

    1. Download pretrained PLY if not available
    2. Convert to .splat format for web
    3. Extract metadata (bounds, Gaussian count)

    Args:
        scene_id: Pretrained scene ID.
        config: Configuration.

    Returns:
        UrbanScene ready for web viewer, or None.
    """
    config = config or SkyfallConfig()

    # Download PLY
    ply_path = download_pretrained_ply(scene_id, config.model_dir)
    if ply_path is None:
        return None

    # Read metadata
    meta = load_ply_metadata(ply_path)

    # Convert to .splat for web
    splat_dir = Path(config.output_dir)
    splat_dir.mkdir(parents=True, exist_ok=True)
    splat_path = ply_to_web_splat(ply_path, splat_dir / f"{scene_id}.splat")

    scene = UrbanScene(
        ply_path=ply_path,
        scene_id=scene_id,
        metadata=meta,
        n_gaussians=meta.get("n_gaussians", 0),
    )

    if splat_path:
        scene.metadata["splat_path"] = str(splat_path)

    return scene
