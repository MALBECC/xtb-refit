# Source this before running any pipeline script:   source code/env.sh
# Activates the conda env and applies the multiprocessing fork+OpenMP fix.
# (common.py also sets the OMP/etemp vars in-process; this additionally activates conda.)
#
# PORTABILITY: override XTBFIT_ENV (env name) or XTBFIT_CONDA (path to conda.sh) for the
# target machine, e.g.:  XTBFIT_ENV=myenv source code/env.sh

: "${XTBFIT_ENV:=xtbfit}"

_activate() {
    # 1) explicit override
    if [ -n "$XTBFIT_CONDA" ] && [ -f "$XTBFIT_CONDA" ]; then
        source "$XTBFIT_CONDA" && conda activate "$XTBFIT_ENV" && return 0
    fi
    # 2) conda/mamba already on PATH
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$XTBFIT_ENV" && return 0
    fi
    # 3) probe common install locations
    for base in "$HOME/mambaforge" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                "/home/jsemelak/Programs/mambaforge" "/opt/conda" "/opt/mambaforge"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            source "$base/etc/profile.d/conda.sh" && conda activate "$XTBFIT_ENV" && return 0
        fi
    done
    return 1
}

if _activate; then
    :
else
    echo "WARN: could not activate conda env '$XTBFIT_ENV'. Set XTBFIT_CONDA=/path/to/conda.sh" >&2
    echo "      (see environment.yml / RUN_ON_HPC.md to create the env)" >&2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1
export MPLBACKEND=Agg
export XTB_ETEMP="${XTB_ETEMP:-500}" XTB_MAXITER="${XTB_MAXITER:-300}"

echo "xtb_refit env ready | conda env=$XTBFIT_ENV | python=$(command -v python3) | XTB_ETEMP=$XTB_ETEMP"
