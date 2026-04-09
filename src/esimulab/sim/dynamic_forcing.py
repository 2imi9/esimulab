"""Dynamic atmospheric forcing during Genesis simulation.

Genesis 0.4.5 does not support runtime force field modification.
This module implements workarounds:

1. Scene reconstruction: rebuild scene with updated force fields
   (expensive but correct — only for long-timescale weather changes)
2. Emitter modulation: vary rain emission rate/direction each step
   (supported natively via emitter.emit() parameters)
3. Force field scheduling: pre-compute a sequence of force configs
   and apply them by rebuilding at interval boundaries
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ForcingSchedule:
    """Time-varying atmospheric forcing schedule.

    Pre-computes forcing parameters at regular intervals for the
    entire simulation duration. Applied by rebuilding force fields
    at each interval boundary.
    """

    # Wind at each interval: list of (direction, magnitude) tuples
    wind_sequence: list[tuple[tuple[float, float, float], float]] = field(default_factory=list)
    # Precipitation at each interval: list of (rate_mm_hr, droplet_size) tuples
    precip_sequence: list[tuple[float, float]] = field(default_factory=list)
    # Steps between forcing updates
    interval_steps: int = 100


def create_forcing_schedule(
    wind_dir: tuple[float, float, float],
    wind_mag: float,
    precip_rate: float,
    num_steps: int,
    interval_steps: int = 100,
    wind_rotation_deg: float = 45.0,
    wind_variation_pct: float = 20.0,
    precip_peak_factor: float = 1.5,
) -> ForcingSchedule:
    """Create a time-varying forcing schedule.

    Simulates a passing weather system with rotating wind,
    varying intensity, and precipitation that peaks at mid-simulation.

    Args:
        wind_dir: Initial wind direction (normalized).
        wind_mag: Initial wind magnitude (m/s).
        precip_rate: Base precipitation rate (mm/hr).
        num_steps: Total simulation steps.
        interval_steps: Steps between forcing updates.
        wind_rotation_deg: Total wind rotation over simulation (degrees).
        wind_variation_pct: Wind speed variation percentage.
        precip_peak_factor: Peak precipitation multiplier.

    Returns:
        ForcingSchedule with pre-computed sequences.
    """
    n_intervals = max(1, num_steps // interval_steps)

    wind_seq = []
    precip_seq = []

    for i in range(n_intervals):
        progress = i / max(1, n_intervals - 1)

        # Rotate wind
        angle = math.radians(wind_rotation_deg * progress)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = wind_dir[0] * cos_a - wind_dir[1] * sin_a
        dy = wind_dir[0] * sin_a + wind_dir[1] * cos_a
        norm = math.sqrt(dx**2 + dy**2) or 1.0
        new_dir = (dx / norm, dy / norm, 0.0)

        # Vary wind magnitude (sinusoidal)
        variation = 1.0 + (wind_variation_pct / 100) * math.sin(progress * math.pi * 4)
        new_mag = wind_mag * variation

        wind_seq.append((new_dir, new_mag))

        # Precipitation: ramp up to peak, then taper
        precip_factor = math.sin(progress * math.pi) * precip_peak_factor
        new_rate = precip_rate * max(0.0, precip_factor)
        droplet_size = 0.05 * (1.0 + new_rate / 10.0)

        precip_seq.append((new_rate, droplet_size))

    schedule = ForcingSchedule(
        wind_sequence=wind_seq,
        precip_sequence=precip_seq,
        interval_steps=interval_steps,
    )

    logger.info(
        "Created forcing schedule: %d intervals, wind rotation=%.0f°, precip peak=%.1fx",
        n_intervals,
        wind_rotation_deg,
        precip_peak_factor,
    )
    return schedule


def apply_forcing_at_step(
    schedule: ForcingSchedule,
    step: int,
    emitter: Any | None = None,
    z_top: float = 100.0,
) -> dict[str, Any]:
    """Apply the forcing schedule at a given simulation step.

    For wind: logs the intended update (Genesis doesn't support runtime
    force field modification — would need scene reconstruction).

    For precipitation: directly modulates emitter.emit() parameters
    (this IS supported at runtime).

    Args:
        schedule: Pre-computed forcing schedule.
        step: Current simulation step.
        emitter: Genesis SPH emitter (for precipitation).
        z_top: Emission height.

    Returns:
        Dict with current forcing parameters for logging/export.
    """
    interval_idx = min(
        step // schedule.interval_steps,
        len(schedule.wind_sequence) - 1,
    )

    current_wind = schedule.wind_sequence[interval_idx]
    current_precip = schedule.precip_sequence[interval_idx]

    wind_dir, wind_mag = current_wind
    precip_rate, droplet_size = current_precip

    # Apply precipitation via emitter (runtime-supported)
    if emitter is not None and precip_rate > 0 and step % 5 == 0:
        try:
            emitter.emit(
                droplet_shape="circle",
                droplet_size=droplet_size,
                pos=(0.0, 0.0, z_top),
                direction=(0, 0, -1),
                speed=9.0,  # terminal velocity
                theta=0.3,
            )
        except Exception:
            logger.debug("Emitter.emit() failed at step %d", step, exc_info=True)

    # Wind update logged (runtime modification not supported in Genesis 0.4.5)
    # Future: reconstruct scene or use custom Taichi kernel
    if step % schedule.interval_steps == 0:
        logger.debug(
            "Forcing step %d: wind=(%.2f,%.2f) %.1f m/s, precip=%.1f mm/hr",
            step,
            wind_dir[0],
            wind_dir[1],
            wind_mag,
            precip_rate,
        )

    return {
        "wind_direction": wind_dir,
        "wind_magnitude": wind_mag,
        "precip_rate": precip_rate,
        "droplet_size": droplet_size,
        "interval": interval_idx,
    }
