"""Physics-Informed Neural Operator (PINO) training utilities.

Combines data loss with PDE residual loss for physics-constrained
surrogate training. Uses PhysicsNeMo PDE modules when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PINOConfig:
    """Configuration for PINO training."""

    pde_weight: float = 0.1  # weight of PDE residual loss
    data_weight: float = 1.0  # weight of data loss
    gradient_method: str = "fdm"  # 'fdm', 'fourier', or 'exact'
    nu: float = 0.01  # kinematic viscosity for Navier-Stokes
    rho: float = 1000.0  # fluid density
    learning_rate: float = 1e-3
    epochs: int = 100


def _try_import_navier_stokes():
    """Import PhysicsNeMo Navier-Stokes PDE module."""
    try:
        from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes

        return NavierStokes
    except ImportError:
        return None


def compute_shallow_water_residual(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    g: float = 9.81,
) -> dict[str, np.ndarray]:
    """Compute shallow water equation residuals using finite differences.

    Shallow water equations (2D):
        dh/dt + d(hu)/dx + d(hv)/dy = 0           (continuity)
        d(hu)/dt + d(hu^2 + gh^2/2)/dx + d(huv)/dy = 0  (x-momentum)
        d(hv)/dt + d(huv)/dx + d(hv^2 + gh^2/2)/dy = 0  (y-momentum)

    Args:
        h: Water depth field (H, W).
        u: x-velocity field (H, W).
        v: y-velocity field (H, W).
        dx: Grid spacing in x (meters).
        dy: Grid spacing in y (meters).
        g: Gravitational acceleration.

    Returns:
        Dict with 'continuity', 'x_momentum', 'y_momentum' residual arrays.
    """
    # Continuity: d(hu)/dx + d(hv)/dy
    hu = h * u
    hv = h * v
    dhu_dx = np.gradient(hu, dx, axis=1)
    dhv_dy = np.gradient(hv, dy, axis=0)
    continuity = dhu_dx + dhv_dy

    # x-momentum: d(hu^2 + gh^2/2)/dx + d(huv)/dy
    flux_xx = h * u**2 + g * h**2 / 2
    flux_xy = h * u * v
    x_mom = np.gradient(flux_xx, dx, axis=1) + np.gradient(flux_xy, dy, axis=0)

    # y-momentum: d(huv)/dx + d(hv^2 + gh^2/2)/dy
    flux_yx = h * u * v
    flux_yy = h * v**2 + g * h**2 / 2
    y_mom = np.gradient(flux_yx, dx, axis=1) + np.gradient(flux_yy, dy, axis=0)

    return {
        "continuity": continuity.astype(np.float32),
        "x_momentum": x_mom.astype(np.float32),
        "y_momentum": y_mom.astype(np.float32),
    }


def pino_loss(
    prediction: Any,
    target: Any,
    config: PINOConfig | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict[str, Any]:
    """Compute PINO loss = data_loss + pde_weight * pde_residual_loss.

    Args:
        prediction: Model output tensor (B, C, H, W).
            Channel 0: water depth (h)
            Channel 1: x-velocity (u) — optional
        target: Ground truth tensor (B, C, H, W).
        config: Training configuration.
        dx: Grid spacing in x.
        dy: Grid spacing in y.

    Returns:
        Dict with 'total', 'data', 'pde' loss values.
    """
    import torch

    config = config or PINOConfig()

    # Data loss (MSE)
    data_loss = torch.nn.functional.mse_loss(prediction, target)

    # PDE residual loss (if prediction has enough channels for SWE)
    pde_loss = torch.tensor(0.0, device=prediction.device)
    if prediction.shape[1] >= 3:
        h = prediction[:, 0]  # water depth
        u = prediction[:, 1]  # x-velocity
        v = prediction[:, 2]  # y-velocity (if available)

        # Finite difference gradients
        dhu_dx = (torch.roll(h * u, -1, dims=-1) - torch.roll(h * u, 1, dims=-1)) / (2 * dx)
        dhv_dy = (torch.roll(h * v, -1, dims=-2) - torch.roll(h * v, 1, dims=-2)) / (2 * dy)
        continuity_residual = dhu_dx + dhv_dy
        pde_loss = torch.mean(continuity_residual**2)

    total = config.data_weight * data_loss + config.pde_weight * pde_loss

    return {
        "total": total,
        "data": data_loss,
        "pde": pde_loss,
    }


def create_navier_stokes_constraint(
    nu: float = 0.01,
    rho: float = 1.0,
    dim: int = 2,
) -> Any | None:
    """Create a PhysicsNeMo Navier-Stokes PDE constraint.

    Args:
        nu: Kinematic viscosity.
        rho: Fluid density.
        dim: Spatial dimension (2 or 3).

    Returns:
        NavierStokes PDE object, or None if PhysicsNeMo unavailable.
    """
    ns_cls = _try_import_navier_stokes()
    if ns_cls is None:
        logger.warning("PhysicsNeMo NavierStokes not available")
        return None

    ns = ns_cls(nu=nu, rho=rho, dim=dim)
    logger.info("Created Navier-Stokes constraint: nu=%f, rho=%f, dim=%d", nu, rho, dim)
    return ns
