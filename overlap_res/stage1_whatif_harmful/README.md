# Fixed-Model Harmful-Overlap What-If

This suite keeps the DeepSeek-derived model dimensions fixed and only changes
runtime overlap pressure:

- GEMM proxy: `m=2048, k=7168, n=4096`
- Optional fused SwiGLU proxy: `n=8192`
- Communication payload: `2048 * 7168 * 2 = 28 MiB`
- Offset grid: `[0.0]`
- Default runtime: `CUDA_DEVICE_MAX_CONNECTIONS=8`, `NCCL_ALGO=Ring`

Submit:

```bash
cd /iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate/overlap_res/stage1_whatif_harmful
sbatch run_whatif_harmful.sh
```

Useful overrides:

```bash
WHATIF_VARIANTS=anchor_ar_simple,ar_simple_ch8 sbatch run_whatif_harmful.sh
WHATIF_INCLUDE_FUSED=1 sbatch run_whatif_harmful.sh
WHATIF_RUN_NSYS=0 sbatch run_whatif_harmful.sh
WHATIF_TIMING_MEASURE_ITERS=50 sbatch run_whatif_harmful.sh
```

The job writes timing rows to:

```text
<jobid>/atlas.jsonl
<jobid>/atlas_summary.json
<jobid>/raw/<variant>/rank_<rank>.jsonl
```

It then selects either the first valid harmful case or the highest-tax valid
overlap case for one Nsight Systems sample under:

```text
<jobid>/nsys/
```

The success condition is:

```text
classification == harmful_overlap
benefit_vs_serial_same_stream < 0
actual_overlap_ms > 0.05
min(pairwise_overlap_pct_a, pairwise_overlap_pct_b) >= 0.30
```

Nsight should be checked using the NVTX ranges:

```text
...::offset_sweep_0::deepseek_*gemm*::offset_sweep
...::offset_sweep_1::tp_hidden_*::offset_sweep
```
