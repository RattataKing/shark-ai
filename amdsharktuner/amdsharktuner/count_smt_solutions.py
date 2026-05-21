#!/usr/bin/env python3
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: python -m amdsharktuner.count_smt_solutions ./dispatch_tuner/dispatch_sample_benchmark_with_constraints_0.mlir > dispatch_tuner/iree_solution_count.log

import argparse
from pathlib import Path

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
import z3  # type: ignore

from amdsharktuner.common import TunerContext
from amdsharktuner.smt_candidate_gen import generate_solutions_from_constraint_op


def _get_constraints_ops(
    module: ir.Module,
) -> list[iree_codegen.ConstraintsOp]:
    ops = ir.get_ops_of_type(module, iree_codegen.ConstraintsOp)
    print(f"Total ConstraintsOp count: {len(ops)}")
    return ops


def _count_solutions_for_op(
    constraints_op: iree_codegen.ConstraintsOp,
    z3_ctx: z3.Context,
    dump_path: Path,
) -> int:
    smtlib = iree_codegen.convert_constraints_op_to_smtlib(
        constraints_op, emit_reset=False
    )
    z3_solver = z3.Solver(ctx=z3_ctx)
    z3_solver.add(z3.parse_smt2_string(smtlib, ctx=z3_ctx))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(z3_solver.to_smt2(), encoding="utf-8")
    print(f"Dumped z3 constraints to: {dump_path}")
    return sum(
        1
        for _ in generate_solutions_from_constraint_op(
            constraints_op=constraints_op,
            z3_ctx=z3_ctx,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an MLIR file, solve each iree_codegen.ConstraintsOp, "
            "and print the solution count per op."
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
    dump_dir = Path(__file__).resolve().parents[1] / "dispatch_tuner"

    with TunerContext() as tuner_ctx:
        module = ir.Module.parse(mlir_text, tuner_ctx.mlir_ctx)
        constraint_ops = _get_constraints_ops(module)
        if not constraint_ops:
            raise RuntimeError("No iree_codegen.ConstraintsOp found in input MLIR.")

        z3_ctx = z3.Context()
        for index, constraints_op in enumerate(constraint_ops):
            pipeline = constraints_op.pipeline
            dump_path = dump_dir / f"iree_constraint_smt_str_{index}.txt"
            print(
                f"ConstraintsOp {index} (pipeline = {pipeline}): "
                "counting solutions..."
            )
            solution_count = _count_solutions_for_op(
                constraints_op=constraints_op,
                z3_ctx=z3_ctx,
                dump_path=dump_path,
            )
            print(f"ConstraintsOp {index} solution count: {solution_count}")


if __name__ == "__main__":
    main()
