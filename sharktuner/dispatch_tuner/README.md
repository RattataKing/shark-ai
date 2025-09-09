**GEMM `compute_gemm_4096_4096_8192_f16_f32_tA.mlir` with 1024 candidates**

From [`tuning_compute_gemm_4096_4096_8192_f16_f32_tA.csv`](https://github.com/RattataKing/shark-ai/blob/dispatch_tuner/sharktuner/dispatch_tuner/single_gemm/compute_gemm_4096_4096_8192_f16_f32_tA/tuning_compute_gemm_4096_4096_8192_f16_f32_tA.csv):

`winners` is defined as:
```python
df["winners"] = (df["benchmark_result_order"] <= 20) & (df["benchmark_speedup"] < 1)
```

Correlation matrix between features `cfg.*` and `winners`:
```
winners                                                     1.000000
cfg.workgroup_tile_sizes_[256, 128, 0]                      0.344806
cfg.workgroup_tile_sizes_[128, 512, 0]                      0.211381
cfg.workgroup_tile_sizes_[128, 256, 0]                      0.189788
cfg.workgroup_tile_sizes_[256, 256, 0]                      0.096038
cfg.wg_x                                                    0.079195
cfg.sg_n_cnt                                                0.079195
cfg.workgroup_tile_sizes_[256, 64, 0]                       0.073947
cfg.workgroup_tile_sizes_[64, 512, 0]                       0.071446
cfg.mma_attr_#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>    0.069226
Name: winners, dtype: float64
```
> Notes: if constant column values -> variance = 0 -> correlation will be defined as `NaN`

Correlation Heatmap (Top 10 Features vs Winners)
![Correlation Heatmap (Top 10 Features vs Winners)](https://github.com/RattataKing/shark-ai/blob/dispatch_tuner/sharktuner/dispatch_tuner/single_gemm/compute_gemm_4096_4096_8192_f16_f32_tA/correlation_heatmap.png)