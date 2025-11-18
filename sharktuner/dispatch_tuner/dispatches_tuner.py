import subprocess
from pathlib import Path
import os
import random
import time
from datetime import datetime

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
PICK_SAMPLE = False 
MAX_SAMPLE_SIZE = 50

def main():
    mlir_folder_path = Path("~/iree-kernel-benchmark/dump_dispatch/problem_mlir_dump").expanduser().resolve()
    mlir_files = sorted(mlir_folder_path.glob("*.mlir"))
    mlir_benchmark_folder_path = Path("~/shark-ai/sharktuner/dispatch_tuner/dump").expanduser().resolve()
    mlir_benchmark_files = sorted(mlir_benchmark_folder_path.glob("*.mlir"))

    print(f"[INFO] Found {len(mlir_files)} file(s) in {mlir_folder_path}")
    print(f"[INFO] Found {len(mlir_benchmark_files)} file(s) in {mlir_benchmark_folder_path}")

    if len(mlir_files) != len(mlir_benchmark_files):
        print("[ERROR] Mismatch between number of MLIR files and benchmark files!")

    if PICK_SAMPLE:
        pick_size = min(max(int(len(mlir_benchmark_files) * 0.3), 1), MAX_SAMPLE_SIZE)
        mlir_benchmark_files = random.sample(mlir_benchmark_files, pick_size)

    failed_files = []
    ok = fail = 0

    # --- timing + logging setup ---
    start_dt = datetime.now()
    start_perf = time.perf_counter()
    log_lines: list[str] = []
    log_lines.append(f"Run started at {start_dt.isoformat(timespec='seconds')}\n")

    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_dir = Path(base_path) / "tuning_database"
    csv_dir.mkdir(exist_ok=True)

    for bench in mlir_benchmark_files:
        file_start = time.perf_counter()

        mlir_filename = bench.stem.replace("_benchmark","")
        mlir = Path(f"{mlir_folder_path}/{mlir_filename}.mlir")

        if not mlir.exists():
            print(f"Can't find {mlir}, skipping")
            fail += 1
            failed_files.append(mlir.name)
            finished_at = datetime.now()
            elapsed = time.perf_counter() - file_start
            log_lines.append(
                f"{finished_at.isoformat(timespec='seconds')} - {mlir.name}: "
                f"MISSING in {elapsed:.2f}s\n"
            )
            continue
        # if (f"tuning_{mlir.stem}.csv" not in redo_list) and ((csv_dir / f"tuning_{mlir.stem}.csv").exists()):
        #     print(f"{mlir.stem} already tuned, skipping...")
        #     ok += 1
        #     continue
        print("\n" + "=" * 80)
        print(f"[INFO] Tuning: {mlir.name} with {bench.name}")
        print("=" * 80)

        cmd = [
            "python3", "-m", "dispatch_tuner",
            str(mlir),
            str(bench),
            "--compile-flags-file=dispatch_tuner/compile_flags.txt",
            "--devices=hip://0",
            "--num-candidates=2048"
        ]

        rc = subprocess.call(cmd, cwd=Path("~/shark-ai/sharktuner").expanduser())
        finished_at = datetime.now()
        elapsed = time.perf_counter() - file_start
        if rc == 0:
            ok += 1
            status = "OK"
        else:
            print(f"[WARN] Failed ({rc}) for {mlir.name}")
            fail += 1
            failed_files.append(mlir.name)
            status = f"FAIL({rc})"

        
        log_lines.append(
            f"{finished_at.isoformat(timespec='seconds')} - {mlir.name}: "
            f"{status} in {elapsed:.2f}s\n"
        )

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
    if failed_files:
        log_lines.append("\nFailed MLIR files:\n")
        for name in failed_files:
            log_lines.append(f"{name}\n")

    failed_log = Path("./dispatch_tuner/failed_mlirs.txt")
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with failed_log.open("w") as f:
        f.writelines(log_lines)

    print(f"\n[INFO] Wrote log to {failed_log.resolve()}")

    print("\n" + "-" * 80)
    print(f"[SUMMARY] Success: {ok} | Fail: {fail}")
    print("[SUMMARY] Each successful run should have written "
          "`tuning_<mlir-name>.csv` in dispatch_tuner.py's folder.")
    print(f"[SUMMARY] Total elapsed: {total_elapsed:.2f}s")
    print("-" * 80)


if __name__ == "__main__":
    main()