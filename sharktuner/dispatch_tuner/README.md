Dir Hierarchy:
```
--iree-build
--iree-kernel-benchmark
--shark-ai
```

**Get Tuning Database**
1. Get problem mlir dispatches, see [LINK](https://github.com/RattataKing/iree-kernel-benchmark/blob/dump_gemm/dump_dispatch/README.md)

2. Get exe-benchmark mlir for each dispatch:
```bash
~/shark-ai/sharktuner
python dispatch_tuner/compile_dump_exe.py # Before run, double check command flags (ex. gfx device) in run_iree_compile()
```
Env:
```
--iree-kernel-benchmark
    |__dump_dispatch
        |__problem_mlir_dump
            |__<dispatch 1>.mlir
            |__<dispatch 2>.mlir
            |__...
--shark-ai
    |__sharktuner
        |__dispatch_tuner
            |__dump
                |__<dispatch 1>_benchmark.mlir
                |__<dispatch 2>_benchmark.mlir
                |__...
```

3. Try single dispatch tuner:
```bash
cd ~/shark-ai/sharktuner
source ~/iree-build/.env && export PYTHONPATH
export PATH="$(realpath ~/iree-build/tools):$PATH"

python3 -m dispatch_tuner ~/iree-kernel-benchmark/dump_dispatch/problem_mlir_dump/compute_gemm_4096_4096_8192_f16_f32_tB_benchmark.mlir \
~/shark-ai/sharktuner/dispatch_tuner/dump/compute_gemm_4096_4096_8192_f16_f32_tB_benchmark.mlir \
    --compile-flags-file=dispatch_tuner/compile_flags.txt \
    --devices=hip://6 --num-candidates=30 \
    --dispatch-tuner-num-dispatch-candidates=30
```

4. Run dipatches tuner and get complete tuning database:
```bash
cd ~/shark-ai/sharktuner
python ./dispatch_tuner/dipatches_tuner.py # Before run, modify flags in cmd[] in main() to use target device
```
Env:
```
--iree-kernel-benchmark
    |__dump_dispatch
        |__problem_mlir_dump
            |__<dispatch 1>.mlir
            |__<dispatch 2>.mlir
            |__...
--shark-ai
    |__sharktuner
        |__dispatch_tuner
            |__dump
            |    |__<dispatch 1>_benchmark.mlir
            |    |__<dispatch 2>_benchmark.mlir
            |    |__...
            |__tuning_database
                |__tuning_<dispatch 1>.csv
                |__tuning_<dispatch 2>.csv
                |__...
```


**To experiment with `sort()`**
1. Run `dispatch_tuner.py` as in step 3 above for wanted dispatch mlir, but with 1024 candidates
2. In `libtuner.py`, inside of `benchmark()` function, add lines before calling `benchmark_candidates()`, there is also an exmaple shuffle sort() written in the comments there.
3. Run `dispatch_tuner.py` again for wanted dispatch mlir, remember to change default output file name!!:
```python
output_csv_name = f"tuning_{args.dispatch_file.stem.removesuffix('_benchmark')}_shuffle.csv" # Naming example
```


**Tuning Database Analysis**
Random Forest Importance:
```base
cd ~/shark-ai/sharktuner
python ./dispatch_tuner/analyze.py
```
```
cfg.workgroup_tile_size_x    0.435824
cfg.workgroup_tile_size_y    0.197537
cfg.reduction_tile_size_3    0.156873
cfg.wg_y                     0.080667
cfg.sg_m_cnt                 0.078707
cfg.wg_x                     0.025449
cfg.sg_n_cnt                 0.024944
cfg.subgroup_size            0.000000
cfg.wg_z                     0.000000
cfg.workgroup_tile_size_z    0.000000
```

`winners` is defined as:
```python
df["winners"] = (df["benchmark_result_order"] <= 10) & (df["benchmark_speedup"] < 1)
```
Selected features: `cfg.*`
```
excluded_list = [
    'cfg.workgroup_tile_sizes', # Use splitted column instead: `cfg.workgroup_tile_size_x`, `cfg.workgroup_tile_size_y`, ...
    'cfg.reduction_tile_sizes',
    'cfg.subgroup_tile_sizes',
    'cfg.promote_operands',
    'cfg.pipeline_options_search_space',

    'cfg.M',
    'cfg.N',
    'cfg.K',

    'cfg.mma_attr', # Str Class, need to do one-hot or label, optionally to exclude here
]
```