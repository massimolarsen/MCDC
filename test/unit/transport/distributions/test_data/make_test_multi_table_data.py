import numpy as np


def make_test_multi_table_data():
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
