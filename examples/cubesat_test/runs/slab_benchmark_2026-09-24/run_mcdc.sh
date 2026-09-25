#!/bin/bash
cd "$(dirname "$0")"
export MCDC_LIB=/Users/massimolarsen/Documents/Repos/MCDC/hdf5lib
PY=/Users/massimolarsen/opt/anaconda3/envs/mcdc-env/bin/python
cat cases.txt | xargs -P 4 -L 1 bash -c 'E=$0 T=$1 N=500 OUT=$2 '"$PY"' slab_mcdc.py --mode=numba > $2.log 2>&1; echo "done $2"'
echo ALL_MCDC_DONE
