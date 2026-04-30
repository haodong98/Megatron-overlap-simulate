#!/bin/bash
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --account=a139
#SBATCH --time=00:40:00
#SBATCH --job-name=onecase_gemm_ar
#SBATCH --output=/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate/overlap_res/stage1_onecase_gemm_ar/slurm-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate/overlap_res/stage1_onecase_gemm_ar/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000
#SBATCH --no-requeue

set -euo pipefail

MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate}
ENV_FILE=${ENV_FILE:-$MEGATRON_LM_DIR/kernel_simulator/ngc_25-11-nemo-alps1-single-gpu.toml}
BASE_OUT=$MEGATRON_LM_DIR/overlap_res/stage1_onecase_gemm_ar
RUN_DIR=$BASE_OUT/${SLURM_JOB_ID}
CASE=$BASE_OUT/gemm_x_all_reduce.yaml
NSYS_CASE=$BASE_OUT/gemm_x_all_reduce_nsys.yaml
MANIFEST=$RUN_DIR/manifest_onecase.json

mkdir -p "$RUN_DIR" "$RUN_DIR/nsys"
cd "$MEGATRON_LM_DIR"

export PYTHONPATH="$MEGATRON_LM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-128}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Stage-1 wants to measure potential overlap/contention, so keep enough CUDA
# work queues by default. Override to 1 only when intentionally reproducing the
# serialized production launch policy from deepseek_128_cap.sh.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}

echo "=== one-case Stage-1 smoke ==="
echo "host=$(hostname)"
echo "run_dir=$RUN_DIR"
echo "case=$CASE"
echo "env=$ENV_FILE"
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
echo "NCCL_ALGO=$NCCL_ALGO"
echo "NCCL_PROTO=$NCCL_PROTO"

python3 - "$CASE" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

import yaml

case_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
case = yaml.safe_load(case_path.read_text())
entry = {
    "pair_id": "stage1_smoke_bf16_gemm_x_all_reduce",
    "section": "onecase_smoke",
    "mode": "offset_sweep",
    "case": case,
    "runtime": case.get("runtime", {}),
    "requires": case.get("requires", {}),
}
manifest = {
    "kind": "stage1_offset_sweep_manifest",
    "entry_count": 1,
    "entries": [entry],
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "=== timing offset_sweep via stage1 manifest ==="
srun -u \
  --mpi=none \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --network=disable_rdzv_get \
  --environment="$ENV_FILE" \
  bash -lc "
    set -euo pipefail
    cd $MEGATRON_LM_DIR
    export PYTHONPATH=$MEGATRON_LM_DIR:\${PYTHONPATH:-}
    export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS
    export NCCL_ALGO=$NCCL_ALGO
    export NCCL_PROTO=$NCCL_PROTO
    torchrun --standalone --nproc_per_node=4 -m kernel_simulator \
      --stage1-run-manifest $MANIFEST
    python3 -m kernel_simulator --stage1-aggregate $RUN_DIR/raw || true
  "

echo "=== nsys offset_sweep sample ==="
srun -u \
  --mpi=none \
  --ntasks=1 \
  --gpus-per-task=4 \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --network=disable_rdzv_get \
  --environment="$ENV_FILE" \
  bash -lc "
    set -euo pipefail
    cd $MEGATRON_LM_DIR
    export PYTHONPATH=$MEGATRON_LM_DIR:\${PYTHONPATH:-}
    export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS
    export NCCL_ALGO=$NCCL_ALGO
    export NCCL_PROTO=$NCCL_PROTO
    if command -v nsys >/dev/null 2>&1; then
      torchrun --standalone --nproc_per_node=4 --no-python \
        nsys profile \
          -t cuda,nvtx,cublas,cudnn \
          --sample=none \
          --cpuctxsw=none \
          --force-overwrite=true \
          -o $RUN_DIR/nsys/gemm_x_all_reduce_rank%q{RANK} \
          python3 -m kernel_simulator \
            --case $NSYS_CASE \
            --mode offset_sweep
    else
      echo 'nsys not found inside the job environment; timing results are still available.'
    fi
  "

echo "=== result files ==="
find "$RUN_DIR" -maxdepth 4 -type f | sort
