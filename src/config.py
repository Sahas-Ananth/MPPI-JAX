import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from jax import Array
from jax import numpy as jnp


@dataclass(frozen=True, kw_only=True, order=False)
class MppiConfig:
    """Config value for MPPI algorithm.

    Attributes:
        NO_ROLLOUTS: Number of Rollouts to be generated
        NO_SAMPLES: Length of the Horizon
        DT: Difference between 2 timesteps in a rollout
        TEMPERATURE: Controls the amount of exploration in the algorithm
        LIN_VEL_VAR: Variance of the Linear Velocity
        ANG_VEL_VAR: Variance of the Angular Velocity
        MIN_LIN_VEL: Minimum Linear Velocity
        MAX_LIN_VEL: Maximum Linear Velocity
        MIN_ANG_VEL: Minimum Angular Velocity
        MAX_ANG_VEL: Maximum Angular Velocity
        VISUALIZE: Visualizations Flag
        CAR_WIDTH: Width of the Car only for visualization
        CAR_HEIGHT: Height of the Car only for visualization
        SHOW_ALL_ROLLOUTS: Flag to visualize all rollouts
        INIT_SEED: Seed for the random number generator
        MAX_PLAN_LOOP: Maximum number of iterations
        GOAL_TOLERANCE: Goal Tolerance
        MAP_RESOLUTION: Resolution of the Map
        OBS_COST: Cost of the Obstacle
        INFLATION_COST: Cost of the Inflation region around the obstacle
    """

    NO_ROLLOUTS: int
    NO_SAMPLES: int
    DT: float

    TEMPERATURE: float

    LIN_VEL_VAR: float
    ANG_VEL_VAR: float
    MIN_LIN_VEL: float
    MAX_LIN_VEL: float
    MIN_ANG_VEL: float
    MAX_ANG_VEL: float

    VISUALIZE: bool
    CAR_WIDTH: float
    CAR_HEIGHT: float
    SHOW_ALL_ROLLOUTS: bool
    INIT_SEED: int
    MAX_PLAN_LOOP: int
    GOAL_TOLERANCE: float
    MAP_RESOLUTION: float
    OBS_COST: int
    INFLATION_COST: int

    # Do not change
    MIN_LIMS: Array = field(init=False)
    MAX_LIMS: Array = field(init=False)
    NOISE_VAR: Array = field(init=False)
    SIGMA_INV: Array = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "MIN_LIMS", jnp.array([self.MIN_LIN_VEL, self.MIN_ANG_VEL])
        )
        object.__setattr__(
            self, "MAX_LIMS", jnp.array([self.MAX_LIN_VEL, self.MAX_ANG_VEL])
        )
        object.__setattr__(
            self, "NOISE_VAR", jnp.array([[self.LIN_VEL_VAR, 0], [0, self.ANG_VEL_VAR]])
        )
        object.__setattr__(self, "SIGMA_INV", jnp.linalg.inv(self.NOISE_VAR))
        return


def read_config_from_file(file: Path) -> MppiConfig:
    """Creates a config values from json file given as input.

    Args:
        file: An instance of `Path` class.

    Raises:
        FileNotFoundError: If the file does not exists or if the path provided is not a file.
        ValueError: If the file provided is not a `json` file.

    Returns:
        A new instance of `Config` class.
    """
    if not file.exists():
        raise FileNotFoundError(f"Given Path File: {file.absolute()} does not exists.")
    if not file.is_file():
        raise FileNotFoundError(f"Given Path File: {file.absolute()} is not a file.")
    if not file.suffix == ".json":
        raise ValueError("The given file is not a json file.")
    cfg: MppiConfig
    with file.open() as f:
        cfg_dict = json.load(f)
        cfg = MppiConfig(**cfg_dict)
    return cfg


def write_config_to_file(file: Path, config: MppiConfig):
    """Write an instance of `Config` class to json.

    Args:
        file: Path of the file as an instance of `Path` class.
        config: Instance of `Config` dataclass that needs to be written as json.

    Raises:
        FileExistsError: If the file already exists.
        IsADirectoryError: If the path provided is a directory.
        ValueError: If `config` provided is not a dataclass.
    """
    if file.exists():
        raise FileExistsError("Given File exists already exists.")
    if file.is_dir():
        raise IsADirectoryError("Given path is a directory.")
    if not dataclasses.is_dataclass(config):
        raise ValueError("given config object is not a dataclass.")

    # Make the folder if it does not exists, else it is ok.
    file.absolute().parent.mkdir(parents=True, exist_ok=True)
    with file.open(mode="w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)
