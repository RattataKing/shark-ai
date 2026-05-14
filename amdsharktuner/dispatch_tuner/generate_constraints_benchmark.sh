#!/usr/bin/env bash

# Compiles a dispatch with VectorDistribute constraints and saves both the
# inserted constraint IR and the final benchmark IR without overwriting files.
# Usage: ./generate_constraints_benchmark.sh [dispatch_sample.mlir].
# Outputs: dispatch_sample_benchmark_with_constraints_<num>.mlir and
# dispatch_sample_benchmark_<num>.mlir.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-"$script_dir/dispatch_sample.mlir"}"

if [[ ! -f "$input" ]]; then
  echo "Input file does not exist: $input" >&2
  exit 1
fi

source "$HOME/iree-feature-build/.env"
export PYTHONPATH
export PATH="$(realpath "$HOME/iree-feature-build/tools"):$PATH"

next_index() {
  local index=0
  local constraints_candidate
  local benchmark_candidate
  while true; do
    constraints_candidate="$script_dir/dispatch_sample_benchmark_with_constraints_${index}.mlir"
    benchmark_candidate="$script_dir/dispatch_sample_benchmark_${index}.mlir"
    if [[ ! -e "$constraints_candidate" && ! -e "$benchmark_candidate" ]]; then
      printf '%s\n' "$index"
      return
    fi
    index=$((index + 1))
  done
}

output_index="$(next_index)"
constraints_output="$script_dir/dispatch_sample_benchmark_with_constraints_${output_index}.mlir"
benchmark_output="$script_dir/dispatch_sample_benchmark_${output_index}.mlir"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/iree_constraints.XXXXXX")"
dump_dir="$tmp_dir/dump"
mkdir -p "$dump_dir"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

set +e
iree-compile "$input" \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx1201 \
  --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false \
  --iree-codegen-llvmgpu-use-vector-distribution=true \
  --iree-codegen-experimental-verify-pipeline-constraints \
  --mlir-print-ir-after=iree-codegen-insert-smt-constraints \
  --iree-hal-dump-executable-files-to="$dump_dir" \
  -o /dev/null \
  2> "$constraints_output"
status=$?
set -e

echo "Wrote constraint IR to: $constraints_output"

if [[ $status -ne 0 ]]; then
  echo "iree-compile failed with status $status." >&2
  exit "$status"
fi

benchmark_dump="$dump_dir/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir"
if [[ ! -f "$benchmark_dump" ]]; then
  echo "Expected benchmark dump does not exist: $benchmark_dump" >&2
  exit 1
fi

cp "$benchmark_dump" "$benchmark_output"
echo "Wrote erased-constraint benchmark IR to: $benchmark_output"
