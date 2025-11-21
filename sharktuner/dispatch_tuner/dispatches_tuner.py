import subprocess
from pathlib import Path
import os
import random
import time
from datetime import datetime
import logging

random.seed(42) 
redo_list = [
# "tuning_unet_gemm_4096_5120_640_f16_f32_tB.csv",
# "tuning_tk_gemm_2048_10240_1280_f16_f32_tB.csv",
# "tuning_unet_gemm_2048_1280_1280_f16_f32_tB.csv",
# "tuning_tk_gemm_8192_5120_640_f16_f32_tB.csv",
# "tuning_unet_gemm_2048_1280_5120_f16_f32_tB.csv",
# "tuning_tk_gemm_2048_1280_1280_f16_f32_tB.csv",
# "tuning_unet_gemm_2048_10240_1280_f16_f32_tB.csv",
# "tuning_unet_gemm_1024_1280_5120_f16_f32_tB.csv",
# "tuning_tk_gemm_2048_1280_5120_f16_f32_tB.csv",
# "tuning_unet_gemm_8192_5120_640_f16_f32_tB.csv",
# "tuning_unet_gemm_1024_1280_1280_f16_f32_tB.csv",
# "tuning_unet_gemm_1024_10240_1280_f16_f32_tB.csv",
]
ok_list = [
    "compute_gemm_4096_4096_8192_i8_i32_tB.mlir",
    "tk_gemm_128_1280_2048_i8_i32_tB.mlir",
]
failed_list = [
    "unet_gemm_4096_5120_640_f32_f32_tB.mlir",
    "square_gemm_2048_2048_2048_i32_i32_tB.mlir",
    "unet_gemm_64_1280_2048_i32_i32_tB.mlir"
]
PICK_SAMPLE = False
MAX_SAMPLE_SIZE = 5


def setup_logging() -> logging.Logger:
    base_dir = Path(os.path.abspath(__file__)).parent
    log_file_name = "tuning.log"
    run_log_path = base_dir / log_file_name

    # Create file handler for logging to a file.
    # file_handler = logging.FileHandler(run_log_path, mode="w")
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.DEBUG)

    # Create stream handler for logging to the console (only warnings and higher).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that dynamically adds [levelname] for ERROR and WARNING.
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return f"{record.message}"
            else:
                return f"[{record.levelname}] {record.message}"

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = CustomFormatter()

    # Set formatters to handlers.
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger.
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger to the lowest level.
        handlers=[file_handler, console_handler],
    )

    return logging.getLogger()


def main():
    logger = setup_logging()
    mlir_folder_path = Path("~/iree-kernel-benchmark/dump_dispatch/problem_mlir_dump").expanduser().resolve()
    logger.debug(f"In MLIR folder {mlir_folder_path}: ")
    mlir_files = sorted(mlir_folder_path.glob("*.mlir"))
    for f in mlir_files:
        logger.debug(f"{f.stem}")


    mlir_benchmark_folder_path = Path("~/shark-ai/sharktuner/dispatch_tuner/dump").expanduser().resolve()
    logger.debug(f"In MLIR_benchmark folder {mlir_benchmark_folder_path}: ")
    mlir_benchmark_files = sorted(mlir_benchmark_folder_path.glob("*.mlir"))
    for f in mlir_benchmark_files:
        logger.debug(f"{f.stem}")

    logger.info(f"Found {len(mlir_files)} mlir file(s)")
    logger.info(f"Found {len(mlir_benchmark_files)} benchmark file(s)")

    if len(mlir_files) != len(mlir_benchmark_files):
        logger.warning("Mismatch between number of MLIR files and benchmark files!")

    # if PICK_SAMPLE:
    #     pick_size = min(max(int(len(mlir_benchmark_files) * 0.3), 1), MAX_SAMPLE_SIZE)
    #     mlir_benchmark_files = random.sample(mlir_benchmark_files, pick_size)
    #     logger.info(f"Tuning file number cap: {pick_size} / {len(mlir_benchmark_files)}")

    failed_files = []
    ok = fail = 0

    # --- timing + logging setup ---
    start_dt = datetime.now()
    start_perf = time.perf_counter()
    logger.debug(f"Tuning started at {start_dt.isoformat(timespec='seconds')}")

    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_dir = Path(base_path) / "tuning_database"
    csv_dir.mkdir(exist_ok=True)

    for bench in mlir_benchmark_files:
        file_start = time.perf_counter()
        logger.debug(f"File {bench} started at {start_dt.isoformat(timespec='seconds')}")

        mlir_filename = bench.stem.replace("_benchmark","")
        if mlir_filename in ok_list:
            logger.debug(f"Skipping file {mlir_filename} in OK list")
            continue
        if mlir_filename in failed_list:
            logger.debug(f"Skipping file {mlir_filename} in failed list")
            continue
        mlir = Path(f"{mlir_folder_path}/{mlir_filename}.mlir")

        if not mlir.exists():
            print(f"Can't find {mlir}, skipping")
            fail += 1
            failed_files.append(mlir.name)
            finished_at = datetime.now()
            elapsed = time.perf_counter() - file_start
            logger.warning(f"{finished_at.isoformat(timespec='seconds')} - {mlir.name}: MISSING in {elapsed:.2f}")
            continue
        # if (f"tuning_{mlir.stem}.csv" not in redo_list) and ((csv_dir / f"tuning_{mlir.stem}.csv").exists()):
        #     print(f"{mlir.stem} already tuned, skipping...")
        #     ok += 1
        #     continue
        logger.info("=" * 80)
        logger.info(f"Tuning: {mlir.name} with {bench.name}")
        logger.info("=" * 80)

        cmd = [
            "python3", "-m", "dispatch_tuner",
            str(mlir),
            str(bench),
            "--compile-flags-file=dispatch_tuner/compile_flags.txt",
            "--devices=hip://0",
            "--num-candidates=4096"
        ]

        rc = subprocess.call(cmd, cwd=Path("~/shark-ai/sharktuner").expanduser())
        finished_at = datetime.now()
        elapsed = time.perf_counter() - file_start
        if rc == 0:
            ok += 1
            status = "OK"
            logger.info(f"{finished_at.isoformat(timespec='seconds')} - {mlir.name}: {status} in {elapsed:.2f}")
        else:
            fail += 1
            status = f"FAIL({rc})"
            failed_files.append(mlir.name)
            logger.warning(f"{finished_at.isoformat(timespec='seconds')} - {mlir.name}: {status} in {elapsed:.2f}")

    # # Save failed MLIRs to a file
    # if failed_files:
    #     failed_log = Path("./dispatch_tuner/failed_mlirs.txt")
    #     with failed_log.open("w") as f:
    #         for name in failed_files:
    #             f.write(name + "\n")
    #     print(f"\n[INFO] Wrote failed MLIR filenames to {failed_log.resolve()}")

    # print("\n" + "-" * 80)
    # print(f"[SUMMARY] Success: {ok} | Fail: {fail}")
    # print("[SUMMARY] Each successful run should have written "
    #       f"`tuning_<mlir-name>.csv` in dispatch_tuner.py's folder.")
    # print("-" * 80)
    # --- summary logging ---
    if failed_files:
        logger.warning(f"Failed MLIR files {len(failed_files)}):")
        for name in failed_files:
            logger.warning(f"- {name}")

    end_perf = time.perf_counter()
    total_elapsed = end_perf - start_perf

    logger.info("-" * 80)
    logger.info(f"SUMMARY: Success: {ok} | Fail: {fail}")
    logger.info(
        "SUMMARY: Each successful run should have written "
        "`tuning_<mlir-name>.csv` in dispatch_tuner.py's folder."
    )
    logger.info(f"SUMMARY: Total elapsed: {total_elapsed:.2f}s")
    logger.info("-" * 80)


if __name__ == "__main__":
    main()