import math

import numpy as np
import pytest

import mcdc.transport.distribution as dist
from mcdc.constant import DISTRIBUTION_N_BODY, INTERPOLATION_LINEAR, PI


class MockRNG:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def lcg(self, _state_container):
        if self._i >= len(self._values):
            raise IndexError("MockRNG depleted")
        value = self._values[self._i]
        self._i += 1
        return value


def _dummy_state():
    return [dict(rng_seed=0)]


def _make_tabulated(values, cdf):
    values = list(values)
    cdf = list(cdf)
    data = np.array(values + cdf, dtype=np.float64)
    table = {
        "value_offset": 0,
        "value_length": len(values),
        "cdf_offset": len(values),
        "cdf_length": len(cdf),
    }
    return table, data


def _make_multi_table():
    # Two incident-energy tables, each with 3 points.
    grid = [1.0, 3.0]
    offsets = [0.0, 3.0]

    values = [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    pdf = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    cdf = [0.0, 0.5, 1.0, 0.0, 0.6, 1.0]

    data = np.array(grid + offsets + values + pdf + cdf, dtype=np.float64)

    idx = 0
    grid_offset = idx
    idx += len(grid)
    offset_offset = idx
    idx += len(offsets)
    value_offset = idx
    idx += len(values)
    pdf_offset = idx
    idx += len(pdf)
    cdf_offset = idx

    multi_table = {
        "grid_offset": grid_offset,
        "grid_length": len(grid),
        "offset_offset": offset_offset,
        "offset_length": len(offsets),
        "value_offset": value_offset,
        "value_length": len(values),
        "pdf_offset": pdf_offset,
        "pdf_length": len(pdf),
        "cdf_offset": cdf_offset,
        "cdf_length": len(cdf),
    }
    return multi_table, data


def _make_table_data_constant(value=1.0):
    x = [0.0, 10.0]
    y = [value, value]
    data = np.array(x + y, dtype=np.float64)
    table = {
        "x_offset": 0,
        "x_length": len(x),
        "y_offset": len(x),
        "y_length": len(y),
        "interpolation": INTERPOLATION_LINEAR,
    }
    return table, data


def _make_kalbach_mann():
    # Two incident-energy tables, each with 3 points.
    grid = [1.0, 3.0]
    offsets = [0.0, 3.0]

    energy_out = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    pdf = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    cdf = [0.0, 0.5, 1.0, 0.0, 0.2, 1.0]

    # Keep R = 0 and A = 1 for deterministic angular sampling.
    precompound = [0.0] * 6
    angular_slope = [1.0] * 6

    data = np.array(
        grid + offsets + energy_out + pdf + cdf + precompound + angular_slope,
        dtype=np.float64,
    )

    idx = 0
    grid_offset = idx
    idx += len(grid)
    offset_offset = idx
    idx += len(offsets)
    energy_out_offset = idx
    idx += len(energy_out)
    pdf_offset = idx
    idx += len(pdf)
    cdf_offset = idx
    idx += len(cdf)
    precompound_offset = idx
    idx += len(precompound)
    angular_slope_offset = idx

    kalbach = {
        "energy_offset": grid_offset,
        "energy_length": len(grid),
        "offset_offset": offset_offset,
        "offset_length": len(offsets),
        "energy_out_offset": energy_out_offset,
        "energy_out_length": len(energy_out),
        "pdf_offset": pdf_offset,
        "pdf_length": len(pdf),
        "cdf_offset": cdf_offset,
        "cdf_length": len(cdf),
        "precompound_factor_offset": precompound_offset,
        "precompound_factor_length": len(precompound),
        "angular_slope_offset": angular_slope_offset,
        "angular_slope_length": len(angular_slope),
    }
    return kalbach, data


def _make_tabulated_energy_angle():
    grid = [1.0, 3.0]
    offsets = [0.0, 3.0]

    energy_out = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    pdf = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    cdf = [0.0, 0.5, 1.0, 0.0, 0.2, 1.0]

    cosine_offsets = [0.0, 3.0]
    cosine = [-1.0, 0.0, 1.0, -0.5, 0.5, 1.0]
    cosine_pdf = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    cosine_cdf = [0.0, 0.5, 1.0, 0.0, 0.3, 1.0]

    data = np.array(
        grid
        + offsets
        + energy_out
        + pdf
        + cdf
        + cosine_offsets
        + cosine
        + cosine_pdf
        + cosine_cdf,
        dtype=np.float64,
    )

    idx = 0
    energy_offset = idx
    idx += len(grid)
    offset_offset = idx
    idx += len(offsets)
    energy_out_offset = idx
    idx += len(energy_out)
    pdf_offset = idx
    idx += len(pdf)
    cdf_offset = idx
    idx += len(cdf)
    cosine_offset__offset = idx
    idx += len(cosine_offsets)
    cosine_offset = idx
    idx += len(cosine)
    cosine_pdf_offset = idx
    idx += len(cosine_pdf)
    cosine_cdf_offset = idx

    table = {
        "energy_offset": energy_offset,
        "energy_length": len(grid),
        "offset_offset": offset_offset,
        "offset_length": len(offsets),
        "energy_out_offset": energy_out_offset,
        "energy_out_length": len(energy_out),
        "pdf_offset": pdf_offset,
        "pdf_length": len(pdf),
        "cdf_offset": cdf_offset,
        "cdf_length": len(cdf),
        "cosine_offset__offset": cosine_offset__offset,
        "cosine_offset__length": len(cosine_offsets),
        "cosine_offset": cosine_offset,
        "cosine_length": len(cosine),
        "cosine_pdf_offset": cosine_pdf_offset,
        "cosine_pdf_length": len(cosine_pdf),
        "cosine_cdf_offset": cosine_cdf_offset,
        "cosine_cdf_length": len(cosine_cdf),
    }
    return table, data


@pytest.fixture
def mock_rng(monkeypatch):
    def _apply(values):
        rng = MockRNG(values)
        monkeypatch.setattr(dist.rng, "lcg", rng.lcg)
        return rng

    return _apply


def test_tabulated_distribution_sample(mock_rng):
    table, data = _make_tabulated([1.0, 3.0, 7.0], [0.0, 0.4, 1.0])
    mock_rng([0.2])

    sample = dist.sample_tabulated.py_func(table, _dummy_state(), data)
    expected = 1.0 + (0.2 - 0.0) * (3.0 - 1.0) / (0.4 - 0.0)

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)


def test_multi_table_distribution_sample(mock_rng):
    multi_table, data = _make_multi_table()
    # xi0 chooses table (xi0 < 0.5 -> table 1), xi1 samples within that table.
    mock_rng([0.3, 0.2])

    sample = dist.sample_multi_table.py_func(2.0, _dummy_state(), multi_table, data)

    # Expected from the second table (values 100..300), first bin.
    expected = 100.0 + (0.2 - 0.0) / 0.01

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)


def test_level_scattering_sample():
    level = {"C1": 1.0, "C2": 0.5}
    sample = dist.sample_level_scattering.py_func(5.0, level)
    assert math.isclose(sample, 2.0, rel_tol=0.0, abs_tol=1e-12)


def test_maxwellian_sample(mock_rng):
    table, data = _make_table_data_constant(1.0)
    mcdc = {"table_data": [table]}
    maxwellian = {"nuclear_temperature_ID": 0, "restriction_energy": 0.0}

    xi1, xi2, xi3 = 0.9, 0.9, 0.0
    mock_rng([xi1, xi2, xi3])

    sample = dist.sample_maxwellian.py_func(2.0, _dummy_state(), maxwellian, mcdc, data)
    expected = -(math.log(xi1) + math.log(xi2) * math.cos(0.5 * PI * xi3) ** 2)

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)


def test_evaporation_sample(mock_rng):
    table, data = _make_table_data_constant(1.0)
    mcdc = {"table_data": [table]}
    evaporation = {"nuclear_temperature_ID": 0, "restriction_energy": 0.0}

    xi1, xi2 = 0.1, 0.2
    mock_rng([xi1, xi2])

    sample = dist.sample_evaporation.py_func(2.0, _dummy_state(), evaporation, mcdc, data)
    w = 2.0
    g = 1.0 - math.exp(-w)
    expected = -math.log((1.0 - g * xi1) * (1.0 - g * xi2))

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)


def test_kalbach_mann_sample(mock_rng):
    kalbach, data = _make_kalbach_mann()

    # xi1 chooses table 1, xi2 samples within it, xi3/xi4 set mu deterministically.
    xi1, xi2, xi3, xi4 = 0.3, 0.1, 0.7, 0.5
    mock_rng([xi1, xi2, xi3, xi4])

    E, mu = dist.sample_kalbach_mann.py_func(2.0, _dummy_state(), kalbach, data)

    # Expected values from analytic reconstruction of the configured tables.
    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    E_new = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)

    assert math.isclose(E, E_new, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_tabulated_energy_angle_sample(mock_rng):
    table, data = _make_tabulated_energy_angle()

    xi1, xi2, xi3 = 0.3, 0.1, 0.25
    mock_rng([xi1, xi2, xi3])

    E, mu = dist.sample_tabulated_energy_angle.py_func(
        2.0, _dummy_state(), table, data
    )

    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    E_new = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)
    mu_expected = -1.0 + (xi3 - 0.0) / 0.5

    assert math.isclose(E, E_new, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, mu_expected, rel_tol=0.0, abs_tol=1e-12)


def test_nbody_sample_correlated(mock_rng):
    table, data = _make_tabulated([2.0, 4.0, 6.0], [0.0, 0.4, 1.0])
    mcdc = {"nbody_distributions": [table]}
    distribution = {"child_type": DISTRIBUTION_N_BODY, "child_ID": 0}

    # First value samples energy, second value samples isotropic cosine.
    mock_rng([0.2, 0.75])

    E, mu = dist.sample_correlated_distribution.py_func(
        2.0, distribution, _dummy_state(), mcdc, data
    )

    expected_E = 2.0 + (0.2 - 0.0) * (4.0 - 2.0) / (0.4 - 0.0)
    expected_mu = 2.0 * 0.75 - 1.0

    assert math.isclose(E, expected_E, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, expected_mu, rel_tol=0.0, abs_tol=1e-12)
