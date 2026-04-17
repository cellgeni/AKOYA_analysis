#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/sp_env"
OUT_DIR="${PROJECT_ROOT}/env_archive"
OUT_TAR="${OUT_DIR}/sp_env.tar.gz"

mkdir -p "${OUT_DIR}"

# Make sure conda-pack is available in whichever conda install you use for tooling.
# Example:
#   conda install -n base -c conda-forge conda-pack

conda-pack -p "${ENV_PREFIX}" -o "${OUT_TAR}"

echo "Packed environment created at: ${OUT_TAR}"