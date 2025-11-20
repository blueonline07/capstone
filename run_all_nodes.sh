#!/bin/zsh

set -euo pipefail

# Default configuration
DATE="${DATE:-$(date +%Y-%m-%d)}"
BATCH_DIR="${BATCH_DIR:-batch}"
CONCURRENCY="${CONCURRENCY:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${DATE}" ]]; then
  echo "DATE must be set (e.g., export DATE=2025-11-13)" >&2
  exit 1
fi

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "Batch directory not found: ${BATCH_DIR}" >&2
  exit 1
fi

echo "Running demo.py for all nodes in ${BATCH_DIR} on ${DATE} with concurrency ${CONCURRENCY}"

find "${BATCH_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | \
  xargs -0 -I{} -P "${CONCURRENCY}" \
    zsh -c '
      node_path="$1"
      python_bin="$2"
      date_arg="$3"
      batch_root="$4"
      node="$(basename "$node_path")"
      "$python_bin" demo.py \
        --node "$node" \
        --date "$date_arg" \
        --batch-root "$batch_root"
    ' _ {} "${PYTHON_BIN}" "${DATE}" "${BATCH_DIR}"

