#!/bin/bash -l
# The 3D fuel problems of the handoff, sweeping the spatial bases (dg1, dg2, cg1, cg2)
# and the patch layouts (J = 8, 32), on the OSU HPC. Run from examples/rmc on the
# submit node:  ./submit_3d.sh
#
# One job per problem, case and layout; each runs RMC with every basis (10 iterations
# + 1 correction pass, 100 histories per spatial unknown, energy bin and patch), and
# the J = 8 job also runs SMC (10^7 histories; SMC does not depend on the layout).
# Outputs: <problem>/output_<case>_J<J>.h5; a job is skipped when its file already
# exists with all its runs. Pin jobs wait for the rod job of the same case and layout:
# both problems share nuclides, grids and patches, so the pin reuses the transfer
# moments the rod computed (cache/). A final 1-core job (merge_3d.py) merges each
# problem's files into output.h5 and makes its plots and fom.md.
#
# LAYOUTS (default "8 32"), BASES (default "dg1,dg2,cg1,cg2"), RANKS (default 16) and
# LIMIT (default 48:00:00) override the sweep, the ranks and the time limit. J = 32
# makes the patch table and the moments much more expensive (about J^2).
# Logs: logs/<job name>-<job id>.out.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/efficiency/env.sh"
cd "$HERE"
mkdir -p logs

LAYOUTS=${LAYOUTS:-"8 32"}
BASES=${BASES:-dg1,dg2,cg1,cg2}
RANKS=${RANKS:-16}
LIMIT=${LIMIT:-48:00:00}

complete() {  # complete FILE GROUP...: the file holds every group
    python - "$@" <<'PY' 2>/dev/null
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    sys.exit(0 if all(group in f for group in sys.argv[2:]) else 1)
PY
}

declare -A job
ids=()
piece() {  # piece NAME DIR CASE J LABELS PREFIX AFTER
    local name=$1 dir=$2 case=$3 J=$4 labels=$5 prefix=$6 after=${7:-}
    local output="output_${case}_J${J}.h5"
    local groups="" label basis
    for label in $labels; do
        for basis in ${BASES//,/ }; do groups+=" rmc/$label/$basis-J$J"; done
        if [ "$J" = 8 ]; then groups+=" smc/$label"; fi
    done
    if complete "$dir/$output" $groups; then
        echo "$name: done ($dir/$output), skipped"
        return
    fi
    local runs=rmc
    if [ "$J" = 8 ]; then runs=rmc+smc; fi
    local dependency=()
    if [ -n "$after" ] && [ -n "${job[$after]:-}" ]; then
        dependency=(--dependency=afterany:${job[$after]})
    fi
    local id
    id=$(sbatch --parsable --partition="$SWEEP_PARTITION" --ntasks="$RANKS" \
        --time="$LIMIT" --job-name="3d-$name" "${dependency[@]}" \
        --export=ALL,EXAMPLE_DIR="$HERE/$dir",SCRIPT=input.py,${prefix}_CASES=$case,${prefix}_LAYOUTS=$J,${prefix}_RUNS=${runs/+/:},${prefix}_BASES=${BASES//,/:},${prefix}_OUTPUT=$output \
        run3d.slurm)
    job[$name]=$id
    ids+=("$id")
    echo "$name: submitted $id"
}

for J in $LAYOUTS; do
    for case in thermal fast; do
        piece "rod-$case-J$J" fuel_rod_3d "$case" "$J" "$case" FUEL_ROD_3D
        piece "pin-$case-J$J" pin_cell_3d "$case" "$J" \
            "$case-lattice $case-isolated" PIN3D "rod-$case-J$J"
    done
done

dependency=()
if [ ${#ids[@]} -gt 0 ]; then
    dependency=(--dependency=afterany:$(IFS=:; echo "${ids[*]}"))
fi
id=$(sbatch --parsable --partition="$SWEEP_PARTITION" --ntasks=1 --time=01:00:00 \
    --job-name=3d-merge "${dependency[@]}" \
    --export=ALL,EXAMPLE_DIR="$HERE",SCRIPT=merge_3d.py run3d.slurm)
echo "merge and plots: submitted $id"
echo "monitor with: squeue -u $USER"
