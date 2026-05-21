#!/usr/bin/env bash

# Compiles a dispatch with constraints for both tile-and-fuse (tf) and
# vector-distribution (vd) pipelines and saves constraint and benchmark IR.
# Usage: ./generate_constraints_benchmark.sh [dispatch.mlir].
# Outputs: <stem>_bench_<tf|vd>_<num>.mlir and
# <stem>_bench_with_constraints_<tf|vd>_<num>.mlir.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-"$script_dir/dispatch_sample.mlir"}"

if [[ ! -f "$input" ]]; then
  echo "Input file does not exist: $input" >&2
  exit 1
fi

input="$(realpath "$input")"
input_dir="$(dirname -- "$input")"
input_stem="$(basename -- "$input" .mlir)"

source "$HOME/iree-feature-build/.env"
export PYTHONPATH
export PATH="$(realpath "$HOME/iree-feature-build/tools"):$PATH"

next_index() {
  local index=0
  local pipeline
  local constraints_candidate
  local benchmark_candidate
  while true; do
    for pipeline in tf vd; do
      constraints_candidate="$input_dir/${input_stem}_bench_with_constraints_${pipeline}_${index}.mlir"
      benchmark_candidate="$input_dir/${input_stem}_bench_${pipeline}_${index}.mlir"
      if [[ -e "$constraints_candidate" || -e "$benchmark_candidate" ]]; then
        index=$((index + 1))
        continue 2
      fi
    done
    printf '%s\n' "$index"
    return
  done
}

compile_pipeline() {
  local pipeline="$1"
  local output_index="$2"
  local constraints_output="$input_dir/${input_stem}_bench_with_constraints_${pipeline}_${output_index}.mlir"
  local benchmark_output="$input_dir/${input_stem}_bench_${pipeline}_${output_index}.mlir"
  local tmp_dir
  local dump_dir
  local benchmark_dump
  local compile_args
  local status

  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/iree_constraints.XXXXXX")"
  dump_dir="$tmp_dir/dump"
  mkdir -p "$dump_dir"

  compile_args=(
    --iree-hal-target-device=hip
    --iree-rocm-target=gfx1201
    --iree-codegen-experimental-verify-pipeline-constraints
    --mlir-print-ir-after=iree-codegen-insert-smt-constraints
    --iree-hal-dump-executable-files-to="$dump_dir"
  )
  if [[ "$pipeline" == "vd" ]]; then
    compile_args+=(
      --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false
      --iree-codegen-llvmgpu-use-vector-distribution=true
    )
  fi

  set +e
  iree-compile "$input" "${compile_args[@]}" -o /dev/null 2> "$constraints_output"
  status=$?
  set -e

  echo "Wrote constraint IR to: $constraints_output"

  if [[ $status -ne 0 ]]; then
    rm -rf "$tmp_dir"
    echo "iree-compile failed for $pipeline pipeline with status $status." >&2
    return "$status"
  fi

  benchmark_dump="$dump_dir/module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir"
  if [[ ! -f "$benchmark_dump" ]]; then
    rm -rf "$tmp_dir"
    echo "Expected benchmark dump does not exist: $benchmark_dump" >&2
    return 1
  fi

  cp "$benchmark_dump" "$benchmark_output"
  rm -rf "$tmp_dir"
  echo "Wrote erased-constraint benchmark IR to: $benchmark_output"
  return 0
}

output_index="$(next_index)"

for pipeline in tf vd; do
  compile_pipeline "$pipeline" "$output_index"
done
