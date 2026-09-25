"""Stream selected Geant4 event diagnostics into the region HDF5 output."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np


INT = "i8"
FLOAT = "f8"
STRING = h5py.string_dtype("utf-8")

SCHEMAS = {
    "E": (
        "event_sv",
        [(name, INT) for name in ("run_id", "event_id", "sv_id")]
        + [(name, FLOAT) for name in (
            "weight", "Edep_total_mev", "Edep_niel_mev", "Edep_ionizing_mev",
            "Eion_primary_mev", "Eion_secondary_mev", "Eion_electron_positron_mev",
            "Eion_proton_mev", "Eion_neutron_mev", "Eion_gamma_mev", "Eion_alpha_mev",
            "Eion_ion_recoil_mev", "Eion_other_mev",
        )],
    ),
    "M": (
        "em_secondary_summary",
        [("run_id", INT), ("event_id", INT)]
        + [(name, kind) for name, kind in (
            ("electron_count", INT), ("electron_creation_energy_mev", FLOAT),
            ("positron_count", INT), ("positron_creation_energy_mev", FLOAT),
            ("gamma_count", INT), ("gamma_creation_energy_mev", FLOAT),
        )],
    ),
    "I": (
        "sv_entry",
        [(name, INT) for name in (
            "run_id", "event_id", "sv_id", "track_id", "parent_id", "pdg",
        )]
        + [(name, FLOAT) for name in (
            "entry_energy_mev", "entry_x_mm", "entry_y_mm", "entry_z_mm",
            "direction_x", "direction_y", "direction_z",
        )]
        + [("creator_process", STRING)]
        + [(name, FLOAT) for name in (
            "vertex_x_mm", "vertex_y_mm", "vertex_z_mm", "vertex_energy_mev",
        )]
        + [("vertex_volume", STRING)],
    ),
    "N": (
        "nuclear_birth",
        [(name, INT) for name in ("run_id", "event_id", "track_id", "parent_id", "pdg")]
        + [(name, FLOAT) for name in (
            "creation_energy_mev", "vertex_x_mm", "vertex_y_mm", "vertex_z_mm",
        )]
        + [("vertex_volume", STRING), ("creator_process", STRING)],
    ),
}


def write_diagnostic_tables(output_path: str | Path, diagnostic_dir: str | Path) -> None:
    with h5py.File(output_path, "a") as file:
        group = file.create_group("seu_diagnostics")
        group.attrs["selection"] = "max_sv_Eion_mev >= diagnostic_min_Eion_mev"
        datasets = {
            code: group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.dtype(fields),
                chunks=(1024,),
                compression="gzip",
                compression_opts=1,
            )
            for code, (name, fields) in SCHEMAS.items()
        }
        buffers = {code: [] for code in SCHEMAS}

        def flush(code: str) -> None:
            rows = buffers[code]
            if not rows:
                return
            dataset = datasets[code]
            values = np.empty(len(rows), dtype=dataset.dtype)
            for i, row in enumerate(rows):
                for (field, _), value in zip(SCHEMAS[code][1], row):
                    values[field][i] = value
            old_size = len(dataset)
            dataset.resize(old_size + len(values), axis=0)
            dataset[old_size:] = values
            rows.clear()

        for path in sorted(Path(diagnostic_dir).glob("worker_*.tsv")):
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.reader(stream, delimiter="\t"):
                    if not row:
                        continue
                    code = row[0]
                    buffers[code].append(row[1:])
                    if len(buffers[code]) == 1024:
                        flush(code)
        for code in SCHEMAS:
            flush(code)
