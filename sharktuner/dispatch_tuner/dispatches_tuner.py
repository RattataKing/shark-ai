import subprocess
from pathlib import Path
import os

def main():
    mlir_folder_path = Path("~/iree-kernel-benchmark/dump_dispatch/problem_mlir_dump").expanduser().resolve()
    mlir_files = sorted(mlir_folder_path.glob("*.mlir"))
    mlir_benchmark_folder_path = Path("~/shark-ai/sharktuner/dispatch_tuner/dump").expanduser().resolve()
    mlir_benchmark_files = sorted(mlir_benchmark_folder_path.glob("*.mlir"))

    print(f"[INFO] Found {len(mlir_files)} file(s) in {mlir_folder_path}")
    print(f"[INFO] Found {len(mlir_benchmark_files)} file(s) in {mlir_benchmark_folder_path}")

    if len(mlir_files) != len(mlir_benchmark_files):
        print("[ERROR] Mismatch between number of MLIR files and benchmark files!")
        return

    failed_files = []
    ok = fail = 0

    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_dir = Path(base_path) / "tuning_database"
    csv_dir.mkdir(exist_ok=True)
    for mlir, bench in zip(mlir_files, mlir_benchmark_files):
        if ((csv_dir / f"tuning_{mlir.stem}.csv").exists()):
            print(f"{mlir.stem} already tuned, skipping...")
            ok += 1
            continue
        print("\n" + "=" * 80)
        print(f"[INFO] Tuning: {mlir.name} with {bench.name}")
        print("=" * 80)

        cmd = [
            "python3", "-m", "dispatch_tuner",
            str(mlir),
            str(bench),
            "--compile-flags-file=dispatch_tuner/compile_flags.txt",
            "--devices=hip://6,hip://7",
            "--num-candidates=1024",
            "--dispatch-tuner-num-dispatch-candidates=1024",
        ]

        rc = subprocess.call(cmd, cwd=Path("~/shark-ai/sharktuner").expanduser())
        if rc == 0:
            ok += 1
        else:
            print(f"[WARN] Failed ({rc}) for {mlir.name}")
            fail += 1
            failed_files.append(mlir.name)

    # Save failed MLIRs to a file
    if failed_files:
        failed_log = Path("./dispatch_tuner/failed_mlirs.txt")
        with failed_log.open("w") as f:
            for name in failed_files:
                f.write(name + "\n")
        print(f"\n[INFO] Wrote failed MLIR filenames to {failed_log.resolve()}")

    print("\n" + "-" * 80)
    print(f"[SUMMARY] Success: {ok} | Fail: {fail}")
    print("[SUMMARY] Each successful run should have written "
          f"`tuning_<mlir-name>.csv` in dispatch_tuner.py's folder.")
    print("-" * 80)

if __name__ == "__main__":
    main()
