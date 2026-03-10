import numpy as np


def make_test_tabulated_data(values, cdf):
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
