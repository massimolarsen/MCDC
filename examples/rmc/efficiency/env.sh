# Site settings for the sweep (sourced by submit.sh and sweep.slurm), for the OSU
# College of Engineering HPC. Checked on the cluster (2026-10-04): the same module stack
# and virtualenv as the MCDC-g4-hpc cubesat jobs (mpi4py 4.1.2 built against the MPICH
# module, numpy 2.4, numba 0.67, h5py, matplotlib; mcdc installed editable from
# hpc-share/MCDC, the feature/RMC checkout).

SHARE=/nfs/hpc/share/$USER
module purge
module load slurm/current python/3.13 mpich/4.0h_gcc-10
source $SHARE/venvs/mcdc-g4/bin/activate

export MCDC_DIR=$SHARE/MCDC                        # feature/RMC checkout
export PYTHONPATH=$MCDC_DIR${PYTHONPATH:+:$PYTHONPATH}
export MCDC_LIB=$MCDC_DIR/hdf5lib_mt5fix           # regenerated library (copied with -L)
export NUMBA_CACHE_DIR=$SHARE/.cache/numba-rmc     # shared kernel cache
export MPLCONFIGDIR=$SHARE/.cache/matplotlib
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Slurm partition used by submit.sh: "share" is open to all users (no account needed).
export SWEEP_PARTITION=share
# ranks, time limits and tasks running at once per problem: see submit.sh
