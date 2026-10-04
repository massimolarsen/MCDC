"""Residual Monte Carlo (RMC) for continuous-energy neutron transport."""

import hashlib
import os
from pathlib import Path


def _set_numba_cache_dir():
    """
    The RMC kernels are compiled with cache=True. Numba invalidates a cached function
    only when its own file changes, not when a callee in another module (other RMC
    modules, mcdc.transport) does, so the cache lives in a directory keyed by a hash of
    every MC/DC source file: any source change starts a fresh cache. An explicit
    NUMBA_CACHE_DIR is respected.
    """
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    import numba

    package = Path(__file__).resolve().parent.parent
    # Skip caches and a virtual environment inside the package directory
    skip = {"__pycache__", "lib", "bin", "share", "include"}
    digest = hashlib.sha256(numba.__version__.encode())
    for directory, subdirectories, files in os.walk(package):
        subdirectories[:] = sorted(d for d in subdirectories if d not in skip)
        for name in sorted(files):
            if name.endswith(".py"):
                path = Path(directory) / name
                digest.update(str(path.relative_to(package)).encode())
                digest.update(path.read_bytes())
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    numba.config.CACHE_DIR = str(root / "mcdc-rmc-numba" / digest.hexdigest()[:16])


_set_numba_cache_dir()
