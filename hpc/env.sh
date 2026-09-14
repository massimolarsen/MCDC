#!/usr/bin/env bash
# Source this file in Bash; do not change the caller's directory or shell options.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source hpc/env.sh' >&2
    exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/stack.sh" || return 1
if [[ ! -x "$MCDC_G4_VENV/bin/python" ]]; then
    echo "Missing virtualenv: $MCDC_G4_VENV. Run bash hpc/bootstrap.sh first." >&2
    return 1
fi
source "$MCDC_G4_VENV/bin/activate" || return 1
if [[ ! -d "$MCDC_LIB" ]]; then
    echo "Missing nuclear data directory: $MCDC_LIB" >&2
    return 1
fi
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" || return 1
echo "MC/DC environment: $VIRTUAL_ENV"
