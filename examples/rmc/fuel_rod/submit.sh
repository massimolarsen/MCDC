#!/bin/bash -l
# Fuel rod: constant vs all-linear trial space vs SMC, both cases (thermal, fast).
# Run from examples/rmc/fuel_rod on the submit node:  ./submit.sh
#
# Each piece is its own job and its own output file. A piece is skipped when its file
# already holds the result (finished runs are not repeated):
#   constant-thermal  input.py  thermal, RMC + SMC  -> output.h5             rmc/thermal smc/thermal
#   constant-fast     input.py  fast, RMC           -> output_fast.h5        rmc/fast
#   smc-fast          input.py  fast, SMC           -> output_smc_fast.h5    smc/fast
#   linear-thermal    linear.py thermal             -> output_linear_thermal.h5
#   linear-fast       linear.py fast                -> output_linear_fast.h5
# A final 1-core job (merge.py) runs when they end: it merges the pieces into output.h5
# and output_linear.h5 and makes the plot.py and plot_linear.py figures.
#
# (sbatch --export splits on commas, so a piece running both RMC and SMC leaves
# FUEL_ROD_RUNS unset.) FUEL_ROD_RANKS (default 16) and FUEL_ROD_LIMIT (default 12:00:00) set each job's ranks
# and time limit. Logs: logs/<job name>-<job id>.out.
set -euo pipefail
export FUEL_ROD_DIR=$(cd "$(dirname "$0")" && pwd)
source "$FUEL_ROD_DIR/../efficiency/env.sh"
cd "$FUEL_ROD_DIR"
mkdir -p logs

RANKS=${FUEL_ROD_RANKS:-16}
LIMIT=${FUEL_ROD_LIMIT:-12:00:00}

has() {  # has FILE GROUP...: the file exists and holds every group
    python - "$@" <<'EOF' 2>/dev/null
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    sys.exit(0 if all(group in f for group in sys.argv[2:]) else 1)
EOF
}

ids=()
piece() {  # piece NAME OUTPUT "GROUPS" EXPORTS
    local name=$1 output=$2 groups=$3 exports=$4
    if has "$output" $groups; then
        echo "$name: done ($output has $groups), skipped"
        return
    fi
    local id
    id=$(sbatch --parsable --partition="$SWEEP_PARTITION" --ntasks="$RANKS" \
        --time="$LIMIT" --job-name="fuel-$name" \
        --export=ALL,FUEL_ROD_DIR="$FUEL_ROD_DIR",FUEL_ROD_OUTPUT="$output",$exports \
        run.slurm)
    ids+=("$id")
    echo "$name: submitted $id"
}

piece constant-thermal output.h5 "rmc/thermal smc/thermal" \
    SCRIPT=input.py,FUEL_ROD_CASES=thermal
piece constant-fast output_fast.h5 "rmc/fast" \
    SCRIPT=input.py,FUEL_ROD_CASES=fast,FUEL_ROD_RUNS=rmc
piece smc-fast output_smc_fast.h5 "smc/fast" \
    SCRIPT=input.py,FUEL_ROD_CASES=fast,FUEL_ROD_RUNS=smc
piece linear-thermal output_linear_thermal.h5 "rmc/thermal" \
    SCRIPT=linear.py,FUEL_ROD_CASES=thermal
piece linear-fast output_linear_fast.h5 "rmc/fast" \
    SCRIPT=linear.py,FUEL_ROD_CASES=fast

dependency=()
if [ ${#ids[@]} -gt 0 ]; then
    dependency=(--dependency=afterany:$(IFS=:; echo "${ids[*]}"))
fi
id=$(sbatch --parsable --partition="$SWEEP_PARTITION" --ntasks=1 --time=01:00:00 \
    --job-name=fuel-merge "${dependency[@]}" \
    --export=ALL,FUEL_ROD_DIR="$FUEL_ROD_DIR",SCRIPT=merge.py run.slurm)
echo "merge and plots: submitted $id"
echo "monitor with: squeue -u $USER"
