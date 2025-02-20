from functools import partial

from jax import Array
from jax import numpy as jnp
from jax import vmap
from jax.experimental.checkify import check
from jax.numpy import linalg


def terminal_cost(x: Array, goal: Array) -> Array:
    """Finds the terminal cost of a rollout by finding the distance (L2 Norm) between the last state of the rollout and the goal.

    Args:
        x: State of the system. (No. of Rollouts x No. of Timesteps x No. of States)
        goal: Goal state of the system. (No. of States)

    Returns:
        Terminal cost of the rollout.
    """
    return linalg.norm(goal - x.at[:, -1, :2].get(), axis=1)


def obstacle_cost(
    x: Array, map_w: Array, resolution: float, origin: tuple[int, int]
) -> Array:
    """Calculates the cost of the current state of the system based on the costmap. The costmap has a high value at the location of the obstacle and a low value everywhere else.

    Args:
        x: One Rollout. (No. of Timesteps x No. of States)
        map_w: Costmap of the environment. (TODO: Add shape here)
        resolution: Resolution of the costmap.
        origin: Origin of the costmap in gridcell.

    Returns:
        Cost of the current rollout.
    """
    check(resolution > 0, "Resolution should be greater than 0.")
    ox, oy = origin
    floor_x = jnp.floor(x.at[:, :2].get() / resolution).astype(jnp.int32)
    return (
        map_w.at[floor_x.at[:, 0].get() + ox, floor_x.at[:, 1].get() + oy].get().sum()
    )


def rollout_cost(
    x: Array,
    u: Array,
    epsilon: Array,
    goal: Array,
    sigma_inv: Array,
    costmap: Array,
    resolution: float,
    origin: tuple[int, int],
    lamb: float,
) -> float:
    """Calculates costs for each step in the rollout. Right now it is: Distance to goal + penalize the result if the new solution is far from the previous
    optimal solution. No. of states is 3 and No. of control inputs is 2. No. of rollouts is K and No. of timesteps is T.

    Args:
        x: State of the system. (No. of Timesteps x No. of States)
        u: Control input to the system. (No. of Timesteps x No. of Control Inputs)
        epsilon: Noise added to the control input. (No. of Timesteps x No. of Control Inputs)
        goal: Goal state of the system. (No. of States)
        sigma_inv: Inverse of the covariance matrix of the noise added to the control input. (No. of Control Inputs x No. of Control Inputs)
        costmap: Costmap of the environment. (TODO: Add shape here)
        resolution: Resolution of the costmap.
        origin: Origin of the costmap in gridcell.
        lamb: Lambda. Taken from the config file.

    Returns:
        Cost of the given rollout.
    """
    c2g = linalg.norm(goal - x.at[:, :2].get(), axis=1).sum()
    keep_prev_ctrl = lamb * (jnp.trace(u @ sigma_inv @ epsilon.T))
    obs_cost = obstacle_cost(x, costmap, resolution, origin)
    return c2g + keep_prev_ctrl + obs_cost


def calculate_costs(
    traj: Array,
    U: Array,
    epsilon: Array,
    goal: Array,
    sigma_inv: Array,
    costmap: Array,
    resolution: float,
    origin: tuple[int, int],
    lamb: float,
) -> Array:
    """Calculates the cost of all the rollouts.

    Calls the `rollout_cost()` for each rollout. Finally, adds the terminal cost to each rollout.

    Args:
        traj (Array): Rollouts of the system. (No. of Rollouts x No. of Timesteps x No. of States)
        U (Array): Control input to the system. (No. of Timesteps x No. of Control Inputs)
        epsilon (Array): Noise added to the control input. (No. of Rollouts x No. of Timesteps x No. of Control Inputs)
        goal (Array): Goal state of the system. (No. of States)
        sigma_inv (Array): Inverse of the covariance matrix of the noise added to the control input. (No. of Control Inputs x No. of Control Inputs)
        costmap (Array): Costmap of the environment.
        resolution (float): Resolution of the costmap.
        origin (Tuple[int, int]): Origin of the costmap in gridcell.
        lamb (float): Lambda. Taken from the config file.

    Returns:
        Array: Cost of the rollouts. (No. of Rollouts x 1)
    """

    f = partial(
        rollout_cost,
        u=U,
        goal=goal,
        sigma_inv=sigma_inv,
        costmap=costmap,
        resolution=resolution,
        origin=origin,
        lamb=lamb,
    )
    costs = vmap(f, in_axes=0)(x=traj, epsilon=epsilon)

    return jnp.add(costs, terminal_cost(traj, goal))
