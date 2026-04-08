"""CLI entry point for Esimulab."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import click


@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box as 'west,south,east,north' in EPSG:4326 degrees.",
)
@click.option(
    "--datetime",
    "dt_str",
    type=str,
    default=None,
    help="ISO datetime for atmospheric data (e.g. '2023-06-15T12:00:00'). Default: now.",
)
@click.option("--steps", type=int, default=600, help="Number of simulation steps.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    help="Output directory for simulation results.",
)
@click.option("--no-gpu", is_flag=True, help="Skip GPU simulation, only fetch data.")
@click.option(
    "--backend",
    type=click.Choice(["gpu", "cpu"], case_sensitive=False),
    default=None,
    help="Genesis compute backend. Default: auto-detect.",
)
@click.option("--serve", is_flag=True, help="Launch web viewer after simulation.")
@click.option("--port", type=int, default=8000, help="Web viewer port.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    ctx: click.Context,
    bbox: str | None,
    dt_str: str | None,
    steps: int,
    output_dir: str,
    no_gpu: bool,
    backend: str | None,
    serve: bool,
    port: int,
    verbose: bool,
) -> None:
    """Esimulab: GPU-accelerated environmental simulation."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # If a subcommand is invoked, skip the default pipeline
    if ctx.invoked_subcommand is not None:
        return

    # Default behavior: run full pipeline (requires --bbox)
    if not bbox:
        click.echo(ctx.get_help())
        return

    logger = logging.getLogger("esimulab")

    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
    except ValueError:
        raise click.BadParameter(
            "Must be 4 comma-separated floats: west,south,east,north"
        ) from None

    time = datetime.fromisoformat(dt_str) if dt_str else datetime.now()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Esimulab starting: bbox=%s, time=%s, steps=%d", bbox_tuple, time, steps)

    from esimulab.pipeline import run_pipeline

    run_pipeline(
        bbox=bbox_tuple,
        time=time,
        num_steps=steps,
        output_dir=output,
        skip_gpu=no_gpu,
        backend=backend,
        serve=serve,
        port=port,
    )


@main.command()
@click.option("--port", type=int, default=8000, help="Web viewer port.")
@click.option("--data-dir", type=click.Path(), default="data", help="Data directory.")
def serve(port: int, data_dir: str) -> None:
    """Launch the web viewer without running a simulation."""
    import uvicorn

    import esimulab.web.server as srv

    srv.DATA_DIR = Path(data_dir)
    click.echo(f"Serving Esimulab viewer at http://localhost:{port}")
    click.echo(f"Data directory: {data_dir}")
    uvicorn.run(srv.app, host="0.0.0.0", port=port)
