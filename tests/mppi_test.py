import numpy as np
import pytest
from jax import jit
from jax.experimental.checkify import checkify

from mppi import calculate_weights


def test_bad_cost_calculate_weights():
    costs = np.array([1, 2, 3, 4, 5]) * -1.0
    err, _ = checkify(jit(calculate_weights, static_argnums=(1,)))(costs, 0.1)
    assert "Costs should be greater than or equal to 0." in str(err.get())


@pytest.mark.parametrize("temperature", [0.0, 1.1])
def test_bad_temp_calculate_weights(temperature: float):
    costs = np.array([1, 2, 3, 4, 5])
    err, _ = checkify(jit(calculate_weights, static_argnums=(1,)))(costs, temperature)
    assert "Temperature must be in (0, 1]." in str(err.get())


def test_good_calculate_weights():
    costs = np.array([0, 1, 2, 3, 4])
    ans = np.array(
        [
            [[8.64703974e-01]],
            [[1.17024957e-01]],
            [[1.58376057e-02]],
            [[2.14338686e-03]],
            [[2.90075868e-04]],
        ]
    )
    err, res = checkify(jit(calculate_weights, static_argnums=(1,)))(costs, 0.5)
    res.block_until_ready()
    assert err.get() is None
    assert np.allclose(res, ans, atol=1e-3)
