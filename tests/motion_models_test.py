import numpy as np
import pytest
from jax import jit
from jax.experimental.checkify import checkify
from numpy.typing import NDArray

from motion_models import simulate, unicycle


def test_bad_ip_state_unicycle():
    with pytest.raises(ValueError) as e:
        _, _ = checkify(jit(unicycle, static_argnums=(2,)))(
            np.zeros((2,)),
            np.zeros((2,)),
            0.1,
        )


def test_bad_ip_control_unicycle():
    with pytest.raises(ValueError) as err:
        _, _ = checkify(jit(unicycle, static_argnums=(2,)))(
            np.zeros((3,)),
            np.zeros((3,)),
            0.1,
        )


def test_bad_ip_dt_unicycle():
    err, _ = checkify(jit(unicycle, static_argnums=(2,)))(
        np.zeros((3,)),
        np.zeros((2,)),
        -0.1,
    )
    assert "Delta time should be greater than 0." in str(err.get())


@pytest.mark.parametrize(
    ("state", "control", "dt", "ans"),
    [
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0]),
            0.1,
            np.array([0.1, 0.0, 0.0]),
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0]),
            0.1,
            np.array([0.1, 0.0, 0.1]),
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0]),
            0.1,
            np.array([0.0, 0.0, 0.1]),
        ),
    ],
)
def test_good_unicycle(
    state: NDArray[np.float64],
    control: NDArray[np.float64],
    dt: float,
    ans: NDArray[np.float64],
):
    err, (res, _) = checkify(jit(unicycle, static_argnums=(2,)))(state, control, dt)
    res.block_until_ready()
    assert err.get() is None
    assert np.allclose(res, ans, atol=1e-3)
