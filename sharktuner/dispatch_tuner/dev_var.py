"""
Do: generate bench stat graph
Note: stddev% = (standard deviation / mean) %
Usage: python dispatch_tuner/dev_var.py
"""

import re
import os
import matplotlib.pyplot as plt
from math import sqrt

PATTERN = re.compile(r"Benchmark time of candidate\s+(\d+):\s*([\d.]+)\s*ms")

input_str = """
2025-09-17 16:09:39,654 - DEBUG - Benchmark time of candidate 1: 497.33 ms
2025-09-17 16:09:43,086 - DEBUG - Benchmark time of candidate 2: 501.67 ms
2025-09-17 16:09:46,017 - DEBUG - Benchmark time of candidate 3: 500.67 ms
2025-09-17 16:09:49,481 - DEBUG - Benchmark time of candidate 4: 497.67 ms
2025-09-17 16:09:52,934 - DEBUG - Benchmark time of candidate 5: 498.33 ms
2025-09-17 16:09:55,861 - DEBUG - Benchmark time of candidate 6: 499.00 ms
2025-09-17 16:09:59,345 - DEBUG - Benchmark time of candidate 7: 501.33 ms
2025-09-17 16:10:01,658 - DEBUG - Benchmark time of candidate 8: 500.33 ms
2025-09-17 16:10:05,121 - DEBUG - Benchmark time of candidate 9: 501.00 ms
2025-09-17 16:10:08,550 - DEBUG - Benchmark time of candidate 10: 498.67 ms
2025-09-17 16:10:11,486 - DEBUG - Benchmark time of candidate 11: 504.33 ms
2025-09-17 16:10:14,918 - DEBUG - Benchmark time of candidate 12: 502.00 ms
2025-09-17 16:10:17,878 - DEBUG - Benchmark time of candidate 13: 507.67 ms
2025-09-17 16:10:21,314 - DEBUG - Benchmark time of candidate 14: 504.67 ms
2025-09-17 16:10:24,285 - DEBUG - Benchmark time of candidate 15: 507.00 ms
2025-09-17 16:10:27,722 - DEBUG - Benchmark time of candidate 16: 505.00 ms
2025-09-17 16:10:31,170 - DEBUG - Benchmark time of candidate 17: 507.33 ms
2025-09-17 16:10:34,102 - DEBUG - Benchmark time of candidate 18: 509.33 ms
2025-09-17 16:10:37,061 - DEBUG - Benchmark time of candidate 19: 509.00 ms
2025-09-17 16:10:40,494 - DEBUG - Benchmark time of candidate 20: 508.33 ms
2025-09-17 16:10:43,430 - DEBUG - Benchmark time of candidate 21: 508.00 ms
2025-09-17 16:10:45,778 - DEBUG - Benchmark time of candidate 22: 507.00 ms
2025-09-17 16:10:49,214 - DEBUG - Benchmark time of candidate 23: 506.67 ms
2025-09-17 16:10:52,902 - DEBUG - Benchmark time of candidate 24: 508.33 ms
2025-09-17 16:10:55,254 - DEBUG - Benchmark time of candidate 25: 506.33 ms
2025-09-17 16:10:58,166 - DEBUG - Benchmark time of candidate 26: 507.33 ms
2025-09-17 16:11:01,066 - DEBUG - Benchmark time of candidate 27: 504.00 ms
2025-09-17 16:11:03,990 - DEBUG - Benchmark time of candidate 28: 507.67 ms
2025-09-17 16:11:06,933 - DEBUG - Benchmark time of candidate 29: 508.67 ms
2025-09-17 16:11:10,358 - DEBUG - Benchmark time of candidate 30: 508.00 ms
2025-09-17 16:11:13,342 - DEBUG - Benchmark time of candidate 31: 509.33 ms
2025-09-17 16:11:16,257 - DEBUG - Benchmark time of candidate 32: 505.00 ms
2025-09-17 16:11:18,534 - DEBUG - Benchmark time of candidate 33: 504.00 ms
2025-09-17 16:11:21,410 - DEBUG - Benchmark time of candidate 34: 503.33 ms
2025-09-17 16:11:24,338 - DEBUG - Benchmark time of candidate 35: 504.67 ms
2025-09-17 16:11:26,614 - DEBUG - Benchmark time of candidate 36: 508.00 ms
2025-09-17 16:11:29,546 - DEBUG - Benchmark time of candidate 37: 504.33 ms
2025-09-17 16:11:31,845 - DEBUG - Benchmark time of candidate 38: 507.33 ms
2025-09-17 16:11:34,741 - DEBUG - Benchmark time of candidate 39: 503.67 ms
2025-09-17 16:11:37,678 - DEBUG - Benchmark time of candidate 40: 507.00 ms
2025-09-17 16:11:39,986 - DEBUG - Benchmark time of candidate 41: 508.33 ms
2025-09-17 16:11:42,901 - DEBUG - Benchmark time of candidate 42: 504.00 ms
2025-09-17 16:11:45,182 - DEBUG - Benchmark time of candidate 43: 503.33 ms
2025-09-17 16:11:48,126 - DEBUG - Benchmark time of candidate 44: 507.67 ms
2025-09-17 16:11:50,438 - DEBUG - Benchmark time of candidate 45: 509.00 ms
2025-09-17 16:11:53,334 - DEBUG - Benchmark time of candidate 46: 507.33 ms
2025-09-17 16:11:56,274 - DEBUG - Benchmark time of candidate 47: 507.00 ms
2025-09-17 16:11:59,166 - DEBUG - Benchmark time of candidate 48: 503.33 ms
2025-09-17 16:12:02,090 - DEBUG - Benchmark time of candidate 49: 509.67 ms
2025-09-17 16:12:05,034 - DEBUG - Benchmark time of candidate 50: 505.33 ms
"""

# --- Extract candidate times ---
pattern = re.compile(r"candidate\s+(\d+):\s*([\d.]+)\s*ms")
pairs = [(int(cid), float(ms)) for cid, ms in pattern.findall(input_str)]
pairs.sort(key=lambda x: x[0])
candidates, times = zip(*pairs)

# --- Stats ---
mean = sum(times) / len(times)
deviations = [t - mean for t in times]
variance = sum(d*d for d in deviations) / len(times)  # population variance
stdev = sqrt(variance)

# --- Plot ---
plt.figure(figsize=(8,4))
plt.plot(candidates, times, marker="o", label="Benchmark time (ms)")

# Mean line
plt.axhline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f} ms")

# ±1 standard deviation bands
plt.axhline(mean + stdev, color="green", linestyle=":", label=f"+1σ = {mean+stdev:.2f} ms")
plt.axhline(mean - stdev, color="green", linestyle=":", label=f"-1σ = {mean-stdev:.2f} ms")

plt.xlabel("Candidate ID")
plt.ylabel("Time (ms)")
plt.title("Benchmark Times with Standard Deviation")
plt.legend()

# Stats box
stats_text = (
    f"Mean = {mean:.2f} ms\n"
    f"Variance = {variance:.4f}\n"
    f"Std Dev = {stdev:.4f} ms"
)
plt.gcf().text(0.75, 0.35, stats_text, fontsize=8, family="monospace",
               bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"))

plt.tight_layout()

# --- Save PNG ---
base_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(base_path, "bench_plot.png")
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved plot -> {out_path}")