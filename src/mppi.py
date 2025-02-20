from functools import partial
from typing import TypeAlias

from jax import Array
from jax import numpy as jnp
from jax import vmap
from jax.experimental.checkify import check

from cost_functions import calculate_costs
from motion_models import simulate, unicycle

MPPIResult: TypeAlias = tuple[Array, Array, Array, Array]


def calculate_weights(costs: Array, temperature: float) -> Array:
    """Calculates the weights of the rollouts.

    The weights are calculated using the costs of the rollouts and the temperature. The
    higher the cost, the lower the weight.

    Args:
        costs: Cost of the rollouts. (No. of Rollouts x 1)
        temperature: Configuration parameter.

    Returns:
        weights of the rollouts. (No. of Rollouts x 1)
    """
    check(jnp.all(costs >= 0), "Costs should be greater than or equal to 0.")
    check(0 < temperature <= 1, "Temperature must be in (0, 1].")
    expo = jnp.exp((-1 * (costs - jnp.min(costs))) / temperature)
    weights = expo / jnp.sum(expo)
    # Expand dims to allow for broadcasting with the control inputs should be the same shape as Epilon (Noise).
    return jnp.expand_dims(weights, axis=(1, 2))


def mppi_plan(
    state: Array,
    U: Array,
    epsilon: Array,
    goal: Array,
    min_lims: Array,
    max_lims: Array,
    sigma_inv: Array,
    costmap: Array,
    resolution: float,
    origin: tuple[int, int],
    temp: float,
    dt: float,
) -> MPPIResult:
    """Pure MPPI planner.

    This is the main function that calls all the other functions to calculate the next control input.

    Args:
        odom: Current state of the robot. (No. of States)
        U: Previous "optimal" control input to the system. (No. of Timesteps x No. of Control Inputs)
        epsilon: Noise added to the control input. (No. of Rollouts x No. of Timesteps x No. of Control Inputs)
        goal: Goal state of the system. (No. of States)
        min_lims: Minimum limits of the control inputs. (No. of Control Inputs)
        max_lims: Maximum limits of the control inputs. (No. of Control Inputs)
        sigma_inv: Inverse of the covariance matrix of the noise added to the control input. (No. of Control Inputs x No. of Control Inputs)
        costmap: Costmap of the environment. (TODO: Add shape here)
        resolution: Resolution of the costmap.
        origin: Origin of the costmap in gridcell.
        temp: temperature. Taken from the config file.
        dt: Delta time. Taken from the config file.

    Returns:
        Next control to apply, Trajectories of the system, Costs of each rollout, Control sequence for the next timestep.
    """
    check(
        U.shape == epsilon.shape[1:],
        "U and epsilon's last 2 dimensions must be the same shape.",
    )
    V = jnp.clip(U + epsilon, min_lims, max_lims)
    clip_eps = V - U
    # TODO: Take partial unicycle function and maybe even simulate as an input to this function.
    trajs = vmap(simulate, in_axes=(None, None, 0))(partial(unicycle, dt=dt), state, V)
    # TODO: Take calulate_costs as an input to this function.
    costs = calculate_costs(
        traj=trajs,
        U=U,
        epsilon=clip_eps,
        goal=goal,
        sigma_inv=sigma_inv,
        costmap=costmap,
        resolution=resolution,
        origin=origin,
        lamb=temp,
    )
    weights = calculate_weights(costs, temp)
    check(
        weights.shape[0] == clip_eps.shape[0],
        "weights and clips_eps leading dimension must be the same.",
    )
    U_next = U + jnp.sum(weights * clip_eps, axis=0)
    u0 = U_next.at[0].get()
    U_next = jnp.roll(U_next, -1, axis=0)
    U_next = U_next.at[-1].set(U_next.at[-2].get())
    return u0, trajs, costs, U_next
