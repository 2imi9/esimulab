# Esimulab - Project Guide

## What is this?
GPU-accelerated environmental simulation platform coupling Genesis physics engine
(terrain, water, soil, wind solvers) with AI-downscaled atmospheric data from
Earth2Studio. Target hardware: NVIDIA RTX 5090 laptop (24GB VRAM).

## Architecture
- `src/esimulab/terrain/` - DEM and land cover data acquisition (dem-stitcher, rasterio)
- `src/esimulab/atmo/` - Atmospheric data fetching (Earth2Studio ERA5/GFS)
- `src/esimulab/sim/` - Genesis physics simulation (SPH water, MPM soil, force-field wind)
- `src/esimulab/web/` - FastAPI server + Three.js/WebGPU web viewer
- `src/esimulab/cli.py` - CLI entry point
- `src/esimulab/pipeline.py` - End-to-end orchestration

## Dev Commands
```bash
uv sync --group dev          # Install all deps including dev tools
uv run pytest                # Run all tests
uv run pytest -m "not gpu"   # Run tests that don't need GPU
uv run ruff check src/       # Lint
uv run ruff format src/      # Format
uv run esimulab --help       # CLI usage
```

## Docker
```bash
docker compose up web-viewer              # Web viewer only (CPU)
docker compose --profile gpu up           # Full stack with GPU
docker compose run genesis-sim python -c "import genesis"  # Test GPU container
```

## Conventions
- Package: `esimulab` (src-layout under `src/esimulab/`)
- Python >=3.11, managed by uv
- Commits: conventional commits (`feat:`, `fix:`, `chore:`, `test:`, `docs:`)
- Branches: GitHub Flow (main + feature branches)
- Tests: pytest with markers `gpu`, `integration`, `slow`
- GPU deps are optional (`[gpu]` extra) - core pipeline works without GPU
- Simulation output goes to `data/` (gitignored, Docker shared volume)

## VRAM Budget (24GB RTX 5090)
Run atmospheric AI inference FIRST, then free weights and start Genesis simulation.
Never load CorrDiff/cBottle and Genesis simultaneously.

## Known Issues (RTX 5090 Blackwell / Windows)
- **Taichi sm_120**: Genesis uses Taichi JIT which may not fully support Blackwell
  (sm_120 compute capability). Workaround: use `--backend cpu` or Docker with
  tested CUDA toolkit. Monitor Taichi releases for Blackwell support.
- **RayTracer**: Requires LuisaRenderPy (Linux-only C++ renderer). On Windows,
  the scene builder automatically falls back to Rasterizer.
- **PyTorch**: Requires cu130 build for native Blackwell support. Install with:
  `pip install torch --index-url https://download.pytorch.org/whl/cu130`
- **CouplerOptions**: Genesis 0.4.5 does not have `gs.options.CouplerOptions`.
  Coupling is automatic via LegacyCouplerOptions/SAPCouplerOptions internally.
