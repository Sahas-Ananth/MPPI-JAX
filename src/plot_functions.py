from jax import Array
from jax import numpy as jnp
from matplotlib import cm as cmx
from matplotlib import colors as colors
from matplotlib import patches as mpatch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# from matplotlib import ticker as mticker

jet = cm = plt.get_cmap("winter_r")


def plot_local_mppi(
    g_pose: Array,
    l_traj: Array,
    costs: Array,
    g_goal: Array,
    l_goal: Array,
    past_pose: Array,
    iteration: int,
    ax: list[Axes],
    CAR_WIDTH: float = 0.75,
    CAR_HEIGHT: float = 0.5,
    SHOW_ALL_ROLLOUTS: bool = False,
):
    # Global Plots
    # Plot past pose i.e. the path taken by the robot
    _ = ax[0].plot(
        past_pose[:, 0], past_pose[:, 1], color="tab:orange", marker="x", linewidth=2
    )

    # Plot global goal and current pose
    _ = ax[0].plot(g_goal[0], g_goal[1], color="r", marker="*", markersize=14)

    # Plot current pose as a rectangle.
    x, y, theta = g_pose
    car = mpatch.Rectangle(
        xy=(x - CAR_WIDTH / 2, y - CAR_HEIGHT / 2),
        width=CAR_WIDTH,
        height=CAR_HEIGHT,
        angle=(theta * 180 / jnp.pi),
        rotation_point="center",
        color="g",
    )
    _ = ax[0].add_patch(car)
    _ = ax[0].plot(0, 0, color="g", marker="o", markersize=14)
    _ = ax[0].set_title(f"Global Frame {iteration}")
    _ = ax[0].grid(linestyle="--", linewidth=0.5)
    _ = ax[0].set_aspect("equal", adjustable="box")

    # Local Plots
    cNorm = colors.Normalize(vmin=costs.min(), vmax=costs.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    if SHOW_ALL_ROLLOUTS:
        for rollout, score in zip(l_traj, costs):
            _ = ax[1].plot(
                rollout[:, 0],
                rollout[:, 1],
                color=scalarMap.to_rgba(score),
                marker="x",
                alpha=0.25,
            )
    best_rollout_index = jnp.argmin(costs)
    _ = ax[1].plot(
        l_traj[best_rollout_index, :, 0],
        l_traj[best_rollout_index, :, 1],
        c="tab:red",
        linewidth=2,
        alpha=0.5,
    )
    _ = ax[1].plot(l_goal[0], l_goal[1], color="r", marker="*", markersize=14)
    _ = ax[1].plot(0, 0, color="g", marker="o", markersize=14)
    _ = ax[1].set_title(f"Local Frame {iteration}")
    _ = ax[1].grid(linestyle="--", linewidth=0.5)
    _ = ax[1].set_aspect("equal", adjustable="box")

    # Plot obstacle
    # obstacle = mpatch.Rectangle((5, 5), 1, 1, color="k")
    # force_field = mpatch.Rectangle((4, 4), 3, 3, color="tab:gray", alpha=0.5)
    # plt.gca().add_patch(force_field)
    # plt.gca().add_patch(obstacle)

    # Plot the legend, title, grid and axis limits
    # gx, gy = goal
    # plt.gca().set_xticks(jnp.arange(-2, gx + 1, 1))
    # plt.gca().set_yticks(jnp.arange(-2, gy + 1, 1))
    # plt.grid(linestyle="--", linewidth=0.5)
    # plt.xlim([-1, gx + 5])
    # plt.ylim([-1, gy + 5])
    # plt.gca().set_aspect("equal", adjustable="box")


def plot_mppi(
    odom: Array,
    traj: Array,
    costs: Array,
    goal: Array,
    past_pose: Array,
    iteration: int,
    CAR_WIDTH: float = 0.75,
    CAR_HEIGHT: float = 0.5,
    SHOW_ALL_ROLLOUTS: bool = False,
):
    cNorm = colors.Normalize(vmin=costs.min(), vmax=costs.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    if SHOW_ALL_ROLLOUTS:
        for rollout, score in zip(traj, costs):
            _ = plt.plot(
                rollout[:, 0],
                rollout[:, 1],
                color=scalarMap.to_rgba(score),
                marker="x",
                alpha=0.25,
            )
    best_rollout_index = jnp.argmin(costs)
    _ = plt.plot(
        traj[best_rollout_index, :, 0],
        traj[best_rollout_index, :, 1],
        c="tab:red",
        linewidth=2,
        alpha=0.5,
    )

    # Plot obstacle
    obstacle = mpatch.Rectangle((5, 5), 1, 1, color="k")
    force_field = mpatch.Rectangle((4, 4), 3, 3, color="tab:gray", alpha=0.5)
    _ = plt.gca().add_patch(force_field)
    _ = plt.gca().add_patch(obstacle)
    #
    # Plot past pose i.e. the path taken by the robot
    _ = plt.plot(
        past_pose[:, 0], past_pose[:, 1], color="tab:orange", marker="x", linewidth=2
    )

    # Plot goal and current pose
    _ = plt.plot(goal[0], goal[1], color="r", marker="*", markersize=14)
    # Plot current pose as a rectangle.
    x, y, theta = odom
    car = mpatch.Rectangle(
        xy=(x - CAR_WIDTH / 2, y - CAR_HEIGHT / 2),
        width=CAR_WIDTH,
        height=CAR_HEIGHT,
        angle=(theta * 180 / jnp.pi),
        rotation_point="center",
        color="g",
    )
    _ = plt.gca().add_patch(car)

    _ = plt.plot(0, 0, color="g", marker="o", markersize=14)

    # Plot the legend, title, grid and axis limits
    # gx, gy = goal
    _ = plt.title(f"MPPI Iteration {iteration}")
    # plt.gca().set_xticks(jnp.arange(-2, gx + 1, 1))
    # plt.gca().set_yticks(jnp.arange(-2, gy + 1, 1))
    plt.grid(linestyle="--", linewidth=0.5)
    # plt.xlim([-1, gx + 5])
    # plt.ylim([-1, gy + 5])
    plt.gca().set_aspect("equal", adjustable="box")


def plot_Us(U: Array):
    _, ax = plt.subplots(2, 2)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    for i, u in enumerate(U):
        ax[0, 0].plot(u[:, 0], label=f"Iteration: {i}")
        ax[0, 1].plot(u[:, 1], label=f"Iteration: {i}")

    for i in range(len(U[0, :, 0])):
        ax[1, 0].plot(U[:, i, 0], label=f"Time: {i}")
        ax[1, 1].plot(U[:, i, 1], label=f"Time: {i}")

    ax[0, 0].set_title("Linear Velocity (time vs velocity)")
    ax[1, 0].set_title("Linear Velocity (iteration vs velocity)")

    ax[0, 1].set_title("Angular Velocity (time vs velocity)")
    ax[1, 1].set_title("Angular Velocity (iteration vs velocity)")


def plot_U0s(U0s: Array, ITER: int):
    """Plots the control command output of the MPPI algorithm.

    Args:
        U0s: History of the control command output
        ITER: No of iterations
    """
    _, ax = plt.subplots(2, 2)
    ax[0, 0].plot(U0s[:, 0], label="Linear Velocity")
    ax[1, 0].plot(U0s[:, 1], label="Angular Velocity")

    ax[0, 0].set_title("Control Input (Linear Velocity)")
    ax[1, 0].set_title("Control Input (Angular Velocity)")
    ax[0, 0].set_xlabel("Iteration")
    ax[1, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Velocity")
    ax[1, 0].set_ylabel("Velocity")

    # Rolling Average Plot.
    ax[0, 1].plot(
        jnp.convolve(U0s[:, 0], jnp.ones(ITER) / ITER, "valid"),
        label="Rolling Average (Linear Velocity)",
    )
    ax[1, 1].plot(
        jnp.convolve(U0s[:, 1], jnp.ones(ITER) / ITER, "valid"),
        label="Rolling Average (Angular Velocity)",
    )
    ax[0, 1].set_title("Avg. Control Input (Linear Velocity)")
    ax[1, 1].set_title("Avg. Control Input (Angular Velocity)")
    ax[0, 1].set_xlabel("Iteration")
    ax[1, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Velocity")
    ax[1, 1].set_ylabel("Velocity")


def plot_times(times: Array, Rollout: int, Horizon: int):
    """Plots the time taken per iteration for the MPPI algorithm.

    Args:
        times: history of the time each iteration
        Rollout: No of rollouts per iteration
        Horizon: No of samples per iteration
    """
    _ = plt.figure()
    _ = plt.semilogy(times)
    _ = plt.title(f"Time taken per iteration for ({Rollout} x {Horizon}) Configuration")
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Time (ms)")
    plt.tight_layout()
    # plt.gca().set_xticks(jnp.arange(0, len(times), 5))
    # plt.gca().yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    # plt.gca().yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
