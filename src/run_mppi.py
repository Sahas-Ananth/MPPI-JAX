import time
from argparse import ArgumentParser
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from jax import Array, jit
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import checkify
from matplotlib import pyplot as plt

from config import MppiConfig, read_config_from_file
from motion_models import unicycle
from mppi import MPPIResult, mppi_plan
from plot_functions import plot_mppi, plot_times


def main(cfg_path: Path) -> None:
    """The main control loop simulating a real robot"""
    # Loading the MPPI Config
    cfg = read_config_from_file(cfg_path)
    # Global Initial state.
    g_pose = jnp.array([0.0, 0.0, 0.0])
    # Global Goal.
    g_goal = jnp.array([10.0, 10.0])
    # Costmap and setting some obstacles.
    costmap = jnp.zeros((10, 10))
    m2gc: Callable[[int], int] = lambda x: int(x / cfg.MAP_RESOLUTION)
    costmap = costmap.at[m2gc(4) : m2gc(7), m2gc(0) : m2gc(7)].set(cfg.INFLATION_COST)
    costmap = costmap.at[m2gc(5) : m2gc(6), m2gc(0) : m2gc(6)].set(cfg.OBS_COST)
    origin: tuple[int, int] = (0, 0)

    # Initial Control.
    u0 = jnp.zeros((2,))
    U_prev = jnp.zeros((cfg.NO_SAMPLES, 2))
    sim: Callable[[Array, Array], tuple[checkify.Error, tuple[Array, Array]]] = (
        checkify.checkify(jit(partial(unicycle, dt=cfg.DT)))
    )
    key = jrand.PRNGKey(cfg.INIT_SEED)
    par_fn: Callable[[Array, Array, Array, Array], MPPIResult] = partial(
        mppi_plan,
        dt=cfg.DT,
        temp=cfg.TEMPERATURE,
        costmap=costmap,
        resolution=cfg.MAP_RESOLUTION,
        origin=origin,
        min_lims=cfg.MIN_LIMS,
        max_lims=cfg.MAX_LIMS,
        sigma_inv=cfg.SIGMA_INV,
    )

    planner: Callable[
        [Array, Array, Array, Array],
        tuple[checkify.Error, MPPIResult],
    ] = checkify.checkify(jit(par_fn))

    # Flags
    fail_flag = True

    history_time_taken = []
    history_pose = []

    i = 0
    for i in range(cfg.MAX_PLAN_LOOP):
        print(
            f"\033[KIteration {i}: Global state: {g_pose}, Goal: {g_goal}, Control: {u0}",
            end="\r",
        )

        if jnp.linalg.norm(g_pose[:2] - g_goal) < cfg.GOAL_TOLERANCE:
            fail_flag = False
            break

        start_time = time.time()
        key, subkey = jrand.split(key)
        noise: Array = jrand.multivariate_normal(
            subkey, np.zeros((2,)), cfg.NOISE_VAR, (cfg.NO_ROLLOUTS, cfg.NO_SAMPLES)
        )
        # state=g_pose, U=U_prev, epsilon=noise, goal=g_goal
        err, (u0, trajs, costs, U_prev) = planner(g_pose, U_prev, noise, g_goal)
        if err.get() is not None:
            raise ValueError(f"\n{err.get() = }")
        _ = u0.block_until_ready()
        _ = trajs.block_until_ready()
        _ = costs.block_until_ready()
        _ = U_prev.block_until_ready()
        time_taken = time.time() - start_time

        history_time_taken.append(time_taken)
        history_pose.append(g_pose.copy())

        err, (g_pose, _) = sim(g_pose, u0)
        if err.get() is not None:
            raise ValueError(f"\n{err.get() = }")

        if costmap[int(g_pose[0]), int(g_pose[1])] > 0:
            print("Collision detected.")
            break

        if cfg.VISUALIZE and (i % 10 == 0 or i == cfg.MAX_PLAN_LOOP - 1):
            plt.clf()
            plot_mppi(
                odom=g_pose,
                traj=trajs,
                costs=costs,
                goal=g_goal,
                past_pose=jnp.array(history_pose),
                iteration=i,
                CAR_WIDTH=cfg.CAR_WIDTH,
                CAR_HEIGHT=cfg.CAR_HEIGHT,
                SHOW_ALL_ROLLOUTS=cfg.SHOW_ALL_ROLLOUTS,
            )
            plt.draw()
            plt.pause(0.5)

    print(
        f"\nReached goal in {i} iterations."
        if not fail_flag
        else "\nFailed to reach goal."
    )
    # Remove the first two elements as they are outliers cause of jit compilation and convert to ms.
    history_time_taken = jnp.array(history_time_taken) * 1000
    print(
        f"Avg Time: {history_time_taken[2:].mean()} ms. Std Dev: {history_time_taken[2:].std()} ms. Min Time: {history_time_taken[2:].min()} ms. Max Time: {history_time_taken[2:].max()} ms."
    )
    print(f"No. of iterations: {i}.")

    # Plotting.
    if cfg.VISUALIZE:
        # Plot the time taken per iteration.
        plot_times(history_time_taken, Rollout=cfg.NO_ROLLOUTS, Horizon=cfg.NO_SAMPLES)
        plt.show()


if __name__ == "__main__":
    # Create an ArgumentParser for parsing CLI arguments.
    parser = ArgumentParser()

    # Create and Parse an argument for the config file to use for this run.
    # This defaults to the 3-link planar RRR robot.
    _ = parser.add_argument(
        "-c",
        "--config_file",
        type=Path,
        default=Path("config/default_MPPI.json"),
        help="Config file path for the program.",
        required=False,
    )
    args = parser.parse_args()

    # from config import write_config_to_file
    #
    # cfg = MppiConfig()
    # write_config_to_file(args.config_file, cfg)

    main(args.config_file)
