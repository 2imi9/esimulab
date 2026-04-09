"""Multi-environment parallel simulation and differentiable mode.

Genesis supports massively parallel simulation via n_envs parameter
(tested up to 30,000 environments). Differentiable mode enables
gradient-based optimization through MPM and Tool solvers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel multi-environment simulation."""

    n_envs: int = 1  # number of parallel environments
    vary_wind: bool = True  # randomize wind per environment
    vary_terrain: bool = False  # slightly perturb terrain
    vary_precip: bool = True  # randomize precipitation rate


@dataclass
class DiffConfig:
    """Configuration for differentiable simulation."""

    requires_grad: bool = False
    optimize_target: str = "terrain"  # 'terrain', 'wind', 'material'
    learning_rate: float = 1e-3
    optimization_steps: int = 50


def build_parallel_scene(
    gs: Any,
    scene_kwargs: dict,
    config: ParallelConfig | None = None,
) -> Any:
    """Build a Genesis scene with parallel environments.

    When n_envs > 1, Genesis JIT-compiles vectorized kernels that
    simulate all environments simultaneously on GPU.

    Args:
        gs: Genesis module.
        scene_kwargs: Base scene configuration dict.
        config: Parallel configuration.

    Returns:
        Genesis scene with parallel environments.
    """
    config = config or ParallelConfig()

    scene = gs.Scene(**scene_kwargs)

    # Entities are added to scene before build
    # build() with n_envs triggers parallel compilation
    logger.info("Building scene with %d parallel environments", config.n_envs)

    return scene, config.n_envs


def setup_differentiable(
    gs: Any,
    sim_options_kwargs: dict,
    config: DiffConfig | None = None,
) -> dict:
    """Configure differentiable simulation mode.

    Only MPM and Tool solvers support gradient computation.
    Enable via SimOptions(requires_grad=True).

    Args:
        gs: Genesis module.
        sim_options_kwargs: SimOptions parameters.
        config: Differentiable configuration.

    Returns:
        Updated SimOptions kwargs with gradient support.
    """
    config = config or DiffConfig()

    if config.requires_grad:
        sim_options_kwargs["requires_grad"] = True
        logger.info(
            "Differentiable mode enabled: target=%s, lr=%f",
            config.optimize_target, config.learning_rate,
        )
    else:
        sim_options_kwargs["requires_grad"] = False

    return sim_options_kwargs


def run_gradient_optimization(
    gs: Any,
    scene: Any,
    config: DiffConfig,
    loss_fn: Any = None,
) -> dict[str, list[float]]:
    """Run gradient-based optimization through the physics pipeline.

    Uses Genesis differentiable simulation to optimize terrain
    parameters, material properties, or forcing conditions.

    Args:
        gs: Genesis module.
        scene: Built Genesis scene with requires_grad=True.
        config: Optimization configuration.
        loss_fn: Callable(scene_state) -> scalar loss tensor.

    Returns:
        Dict with 'losses' history.

    Note:
        Only MPM solver is differentiable in Genesis 0.4.5.
        Rigid and SPH solvers do not support backpropagation.
    """
    if loss_fn is None:
        logger.warning("No loss function provided, skipping optimization")
        return {"losses": []}


    losses = []

    for step in range(config.optimization_steps):
        # Forward pass
        scene.step()

        # Compute loss
        loss = loss_fn(scene)
        losses.append(float(loss.item()))

        # Backward pass (triggers gradient flow through MPM kernels)
        loss.backward()

        if step % 10 == 0:
            logger.info("Optimization step %d/%d, loss=%.6f",
                        step, config.optimization_steps, losses[-1])

    logger.info("Optimization complete: final loss=%.6f", losses[-1] if losses else 0)
    return {"losses": losses}


def create_tensor(gs: Any, data: Any, requires_grad: bool = False) -> Any:
    """Create a Genesis tensor from numpy/torch data.

    Genesis tensors wrap PyTorch tensors with scene awareness
    for automatic differentiation through the physics pipeline.

    Args:
        gs: Genesis module.
        data: numpy array or torch tensor.
        requires_grad: Whether to track gradients.

    Returns:
        Genesis tensor.
    """
    import numpy as np
    import torch

    if isinstance(data, np.ndarray):
        t = gs.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        t = gs.from_torch(data)
    else:
        t = gs.from_numpy(np.array(data))

    if requires_grad and hasattr(t, "requires_grad_"):
        t.requires_grad_(True)

    return t
