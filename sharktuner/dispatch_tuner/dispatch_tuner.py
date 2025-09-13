# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import argparse
import shutil
from pathlib import Path
from sharktuner import libtuner
from sharktuner import common

import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dataclasses import dataclass, asdict, fields
import hashlib
import os
from typing import Any


def dataclass_to_arrow(records: list[Any]) -> pa.Table:
    """
    Convert a list of dataclass instances to a PyArrow Table.
    Supports nested dataclasses by recursively converting them to dicts.
    """
    if not records:
        raise ValueError("No records to convert")

    def serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {f.name: serialize(getattr(obj, f.name)) for f in fields(obj)}
        elif isinstance(obj, list):
            return [serialize(x) for x in obj]
        else:
            return obj

    dicts = [serialize(r) for r in records]
    return pa.Table.from_pylist(dicts)


def write_parquet(records: list[Any], filename: str):
    """Write dataclass records to parquet in the same folder as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, filename)
    table = dataclass_to_arrow(records)
    pq.write_table(table, out_path)
    print(f"Wrote {len(records)} records to {out_path}")



class DispatchTuner(libtuner.TuningClient):
    def __init__(self, tuner_context: common.TunerContext):
        super().__init__(tuner_context)
        self.compile_flags: list[str] = []
        self.benchmark_flags: list[str] = []
        self.compile_timeout: int = 16
        self.benchmark_timeout: int = 16

    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    def get_iree_compile_timeout_s(self) -> int:
        return self.compile_timeout

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return self.benchmark_flags

    def get_benchmark_timeout_s(self) -> int:
        return self.benchmark_timeout


def read_flags_file(flags_file: str) -> list[str]:
    if not flags_file:
        return []

    with open(flags_file) as file:
        return file.read().splitlines()


import csv

def export_to_csv(objects, filename="export.csv"):
    if not objects:
        return None

    rows = []
    headers = []

    for obj in objects:
        row = {}
        for k, v in vars(obj).items():
            if hasattr(v, "__dict__"):  # nested object → flatten one level
                nested = vars(v)
                if nested:  # only if it has attrs
                    for nk, nv in nested.items():
                        key = f"{k}.{nk}"
                        row[key] = nv
                        if key not in headers:
                            headers.append(key)
                else:
                    # skip empty nested object entirely
                    continue
            else:
                row[k] = v
                if k not in headers:
                    headers.append(k)
        rows.append(row)

    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_dir = Path(base_path) / "tuning_database"
    csv_dir.mkdir(exist_ok=True)
    # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunning_database")
    path = os.path.join(csv_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return path



def arg_parse() -> argparse.Namespace:
    # Custom arguments for the example tuner file.
    parser = argparse.ArgumentParser(description="Autotune sample script")
    client_args = parser.add_argument_group("Shark Tuner Options")
    client_args.add_argument(
        "dispatch_file", type=Path, help="Path to the dispatch file to tune (.mlir)"
    )
    client_args.add_argument(
        "--dispatch-tuner-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for dispatch benchmarks.",
    )
    client_args.add_argument(
        "--compile-flags-file",
        type=str,
        default="",
        help="Path to the flags file for iree-compile.",
    )
    client_args.add_argument(
        "--output-td-spec",
        type=Path,
        help="Path to write the best tuned spec. Dumps the best tuned dispatch spec by default, and the best tuned dispatch spec when --stop-after is set to 'benchmark-dispatches'.",
        default="tuning-spec.mlir",
    )
    client_args.add_argument(
        "--dispatch-benchmark-timeout-mins",
        type=float,
        default=None,
        help="Time budget in minutes for disptach benchmark phase.",
    ),
    # Remaining arguments come from libtuner
    args = libtuner.parse_arguments(parser)
    return args

import re
import subprocess

def main() -> None:
    args = arg_parse()

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    stop_after_phase: str = args.stop_after

    print("[WARNING] SHARK Tuner is still experimental")
    root_logger = libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    compile_flags: list[str] = read_flags_file(args.compile_flags_file)

    summary_log_file = path_config.base_dir / "summary.log"
    summary_handler = logging.FileHandler(summary_log_file)
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    try:
        arch = subprocess.check_output("rocminfo | grep gfx", shell=True, text=True)
        match = re.search(r"gfx[0-9a-z]+", arch)
        if match:
            arch = match.group(0)
    except:
        arch = "gfx unknown"

    print("Generating candidate tuning specs...")
    with common.TunerContext(logger=root_logger) as tuner_context:
        tuner_context.logger.addHandler(summary_handler)
        dispatch_tuner = DispatchTuner(tuner_context)
        candidates = libtuner.generate_candidate_specs(args, path_config, dispatch_tuner)
        # print(dispatch_tuner.candidate_records)
        print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")

        print("Compiling dispatch candidates...")
        dispatch_tuner.compile_flags = compile_flags + [
            "--compile-from=executable-sources"
        ]

        dispatch_tuner.benchmark_flags = ["--input=1", "--benchmark_repetitions=3"]
        

        for i in range(len(candidates)):
            dispatch_tuner.tuning_records.append(libtuner.TuningRecord(
                dispatch_id=args.dispatch_file.stem.removesuffix('_benchmark'),
                candidate_id=i,
                op_kind="contraction",
                device = f"{os.uname().nodename}",
                arch = f"{arch}",                                    # gfx942, gfx1100, ...
                cfg=dispatch_tuner.candidate_records[i].cfg,
                compile_flags=dispatch_tuner.compile_flags,
                compile_order_in_list = -1,
                compile_status=False,
                # dispatch_compile_error_class: Optional[str]  # optional: codegen_error, verifier, etc.
                benchmark_flags=dispatch_tuner.get_iree_benchmark_module_flags(),
                to_benchmark = False,
                benchmark_order_in_list=None,
                benchmark_device_id=None,
                benchmark_status=False,
                baseline_benchmark_time_ms=None,
                benchmark_time_ms=None,
                benchmark_result_order=None,
                benchmark_speedup=None,
                benchmark_optimal_time_ms=None,
                benchmark_optimal_start_order=None,
                benchmark_start_order_vs_result=None,
                benchmark_avg_order_diff=None,
                benchmark_avg_top10_order_diff=None,
                candidate_spec_mlir=dispatch_tuner.candidate_trackers[i].spec_path,
            ))

        
        # for i in dispatch_tuner.candidate_records:
        #     i.dispatch_id = args.dispatch_file.stem,
        #     i.device = f"sharkmi300x-4::{args.devices}"
        #     i.arch = "gfx942",
        
        compiled_candidates = libtuner.compile(
            args, path_config, candidates, dispatch_tuner
        )


        message = "Benchmarking compiled dispatch candidates..."
        print(message)
        logging.info(message)
        # dispatch_tuner.benchmark_flags = ["--input=1", "--benchmark_repetitions=3"]
        top_candidates = libtuner.benchmark(
            args,
            compiled_candidates,
            dispatch_tuner,
            args.dispatch_tuner_num_dispatch_candidates,
            args.dispatch_benchmark_timeout_mins,
        )
        logging.info(f"Top dispatch candidates: {top_candidates}")
        for id in top_candidates:
            logging.info(f"{dispatch_tuner.candidate_trackers[id].spec_path.resolve()}")

        print("Check the detailed execution logs in:")
        print(path_config.run_log.resolve())
        print("Check the summary in:")
        print(summary_log_file.resolve())

        # write_parquet(dispatch_tuner.candidate_records, "candidates.parquet")
        # write_parquet(dispatch_tuner.tuning_records, "tuning.parquet")
        # export_to_csv(dispatch_tuner.candidate_records, "candidates.csv")
        # export_to_csv(dispatch_tuner.tuning_records, "tuning.csv")

        # export_to_csv(dispatch_tuner.tuning_records, f"tuning_{args.dispatch_file.stem}.csv")

        # Name CSV by the input file: tuning_<mlir-name>.csv
        output_csv_name = f"tuning_{args.dispatch_file.stem.removesuffix('_benchmark')}.csv"
        csv_path = export_to_csv(dispatch_tuner.tuning_records, output_csv_name)
        print(f"Wrote CSV: {csv_path}")

        # exit()
