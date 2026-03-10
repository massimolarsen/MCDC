import numpy as np


def make_test_tabulated_energy_angle_data():
    grid = [1.0, 3.0, 5.0]
    offsets = [0.0, 3.0, 6.0]

    energy_out = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
    pdf = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    cdf = [0.0, 0.5, 1.0, 0.0, 0.2, 1.0]

    cosine_offsets = [0.0, 3.0, 6.0]
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
