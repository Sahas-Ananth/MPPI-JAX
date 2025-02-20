from collections.abc import Callable
from typing import Any, TypeVar

from jax import Array
from jax import numpy as jnp
from jax.experimental.checkify import check
from jax.lax import scan


def unicycle(state: Array, control: Array, dt: float) -> tuple[Array, Array]:
    """Unicycle motion model.

    Follows the JAX scan function signature

    Args:
        state: A single current state (x, y, theta). Shape: (3,)
        control: A single control input (v, w). Shape: (2,)
        dt: Delta time to simulate the system.

    Returns:
        next_state: Next state of the system. Shape: (3,)
    """
    check(state.shape == (3,), "State should be of shape (3,)")
    check(control.shape == (2,), "Control should be of shape (2,)")
    check(dt > 0, "Delta time should be greater than 0.")
    x: float
    y: float
    theta: float
    v: float
    w: float
    x, y, theta = state
    v, w = control
    x_new: Array = x + v * jnp.cos(theta) * dt
    y_new: Array = y + v * jnp.sin(theta) * dt
    theta_new: float = theta + w * dt
    next_state: Array = jnp.array([x_new, y_new, theta_new])
    return next_state, next_state


Carry = TypeVar("Carry")


def simulate(
    model_fn: Callable[..., tuple[Carry, Array]], initial_state: Array, ctrl_seq: Array
) -> Array:
    """Wraps around motion model and scans through the control sequence to simulate the system for one rollout.

    Args:
        initial_state: Starting state of the Rollout.
        ctrl_seq: Control sequence to simulate the system.
        dt: Delta time to simulate the system.

    Returns:
        Trajectory of the system for one rollout.
    """
    _, answer = scan(model_fn, initial_state, ctrl_seq)
    return answer
