#!/usr/bin/env python3
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: python -m amdsharktuner.count_smt_solutions ./dispatch_tuner/dispatch_sample_benchmark_with_constraints_0.mlir > dispatch_tuner/iree_solution_count.log

import argparse
from collections.abc import Iterator
from pathlib import Path

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
import z3  # type: ignore

from amdsharktuner.common import TunerContext
from amdsharktuner.smt_candidate_gen import generate_solutions_from_constraint_op


def _first_constraints_op(
    module: ir.Module,
) -> Iterator[iree_codegen.ConstraintsOp]:
    ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
    print(f"Total ConstraintsOp count: {len(ops)}")
    if not ops:
        return
    yield ops[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an MLIR file, select the first iree_codegen.ConstraintsOp, "
            "solve it, and print the solution count."
        )
    )
    parser.add_argument("mlir_path", type=Path, help="Path to input MLIR file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlir_path = args.mlir_path
    if not mlir_path.is_file():
        raise FileNotFoundError(f"MLIR file not found: {mlir_path}")

    mlir_text = mlir_path.read_text(encoding="utf-8")
    with TunerContext() as tuner_ctx:
        module = ir.Module.parse(mlir_text, tuner_ctx.mlir_ctx)
        constraints_op = next(_first_constraints_op(module), None)
        if constraints_op is None:
            raise RuntimeError("No iree_codegen.ConstraintsOp found in input MLIR.")

        z3_ctx = z3.Context()
        smtlib = iree_codegen.convert_constraints_op_to_smtlib(
            constraints_op, emit_reset=False
        )
        z3_solver = z3.Solver(ctx=z3_ctx)
        z3_solver.add(z3.parse_smt2_string(smtlib, ctx=z3_ctx))
        dump_path = (
            Path(__file__).resolve().parents[1]
            / "dispatch_tuner"
            / "iree_constraint_smt_str.txt"
        )
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(z3_solver.to_smt2(), encoding="utf-8")
        print(f"Dumped z3 constraints to: {dump_path}")
        solution_count = sum(
            1
            for _ in generate_solutions_from_constraint_op(
                constraints_op=constraints_op,
                z3_ctx=z3_ctx,
            )
        )
        print(f"Solution count: {solution_count}")


if __name__ == "__main__":
    main()
