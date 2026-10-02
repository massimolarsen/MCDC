#!/bin/bash
# Same cases and N as ../slab_benchmark_2026-09-28, run against PR #548 (upstream-dev API).
# 10 MeV is left out: it was cancelled at ~31% on 09-28 (too slow).
cd "$(dirname "$0")"
export MCDC_LIB=/Users/massimolarsen/Documents/Repos/MCDC/hdf5lib
# Unpacked copy of the PR head (git archive pr548); see mcdc-revision.txt
export PYTHONPATH=/private/tmp/claude-501/-Users-massimolarsen-Documents-Repos-MCDC/117b3dbe-e66f-4e7b-bd96-394a48e0d5e2/scratchpad/pr548_src
PY=/Users/massimolarsen/opt/anaconda3/envs/mcdc-env/bin/python
cat cases.txt | xargs -P 4 -L 1 bash -c 'E=$0 T=$1 N=1000 OUT=$2 '"$PY"' slab_mcdc.py --mode=numba > $2.log 2>&1; echo "done $2 $(date +%T)"'
echo ALL_MCDC_DONE
