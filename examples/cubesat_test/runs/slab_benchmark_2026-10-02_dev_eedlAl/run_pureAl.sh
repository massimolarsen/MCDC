#!/bin/bash
# Same cases and N as ../slab_benchmark_2026-10-01_pr548, on merged upstream dev (a0ddec22),
# with Al electron data swapped for the EEDL-based Al.h5 from mcdc-regression_test_data
# (the library the Lockwood regression uses). Mg and Si stay EPRDATA14 from hdf5lib.
cd "$(dirname "$0")"
export MCDC_LIB=/private/tmp/claude-501/-Users-massimolarsen-Documents-Repos-MCDC/117b3dbe-e66f-4e7b-bd96-394a48e0d5e2/scratchpad/lib_eedl_al
export PYTHONPATH=/private/tmp/claude-501/-Users-massimolarsen-Documents-Repos-MCDC/117b3dbe-e66f-4e7b-bd96-394a48e0d5e2/scratchpad/dev_a0ddec22
PY=/Users/massimolarsen/opt/anaconda3/envs/mcdc-env/bin/python
cat cases_pureAl.txt | xargs -P 4 -L 1 bash -c 'PURE_AL=1 E=$0 T=$1 N=1000 OUT=$2 '"$PY"' slab_mcdc.py --mode=numba > $2.log 2>&1; echo "done $2 $(date +%T)"'
echo ALL_MCDC_DONE
