"""Combine MC/DC neutron and proton nuclide files for coupled transport.

MC/DC reads both particles' data from one ``<nuclide>-<T>K.h5`` file in
``$MCDC_LIB``. Proton sources bank secondary neutrons, so a proton run with
neutron transport needs files that hold both. This copies each neutron file
and adds the proton groups from the matching proton file.

TENDL proton data is tabulated at 0 K; it is reused unchanged at the neutron
file's temperature (proton-nucleus cross sections have no meaningful thermal
broadening).

Usage:
    python merge_neutron_data.py NEUTRON_LIB PROTON_LIB OUTPUT_DIR \\
        --temperature 293.6 --nuclides Al27 Si28 O16
"""

import argparse
import os
import shutil

import h5py

PROTON_ITEMS = ["proton_reactions", "secondary_particles", "stopping_power", "radiation_length"]


def merge(nuclide, neutron_lib, proton_lib, output_dir, temperature, proton_temperature):
    neutron_path = os.path.join(neutron_lib, f"{nuclide}-{temperature}K.h5")
    proton_path = os.path.join(proton_lib, f"{nuclide}-{proton_temperature}K.h5")
    output_path = os.path.join(output_dir, f"{nuclide}-{temperature}K.h5")

    shutil.copyfile(neutron_path, output_path)
    with h5py.File(proton_path, "r") as proton, h5py.File(output_path, "a") as output:
        for name in PROTON_ITEMS:
            if name in proton:
                proton.copy(proton[name], output, name=name)
        output.attrs["proton_source_title"] = proton.attrs.get("source_title", "")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("neutron_lib")
    parser.add_argument("proton_lib")
    parser.add_argument("output_dir")
    parser.add_argument("--temperature", default="293.6")
    parser.add_argument("--proton-temperature", default="0.0")
    parser.add_argument("--nuclides", nargs="+", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for nuclide in args.nuclides:
        path = merge(
            nuclide,
            args.neutron_lib,
            args.proton_lib,
            args.output_dir,
            args.temperature,
            args.proton_temperature,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
