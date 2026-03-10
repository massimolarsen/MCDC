import numpy as np


def make_test_kalbach_mann_data():
    # Two incident-energy tables, each with 3 points.
    grid = [1.0, 3.0, 5.0]
    offsets = [0.0, 3.0, 6.0]

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
