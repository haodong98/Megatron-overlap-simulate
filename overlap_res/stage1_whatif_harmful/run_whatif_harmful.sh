#!/bin/bash
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --account=a139
#SBATCH --time=02:00:00
#SBATCH --job-name=whatif_harmful
#SBATCH --output=/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate/overlap_res/stage1_whatif_harmful/slurm-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate/overlap_res/stage1_whatif_harmful/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000
#SBATCH --no-requeue

set -euo pipefail

MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/haodongz/Megatron-overlap-simulate}
ENV_FILE=${ENV_FILE:-$MEGATRON_LM_DIR/kernel_simulator/ngc_25-11-nemo-alps1-single-gpu.toml}
BASE_OUT=$MEGATRON_LM_DIR/overlap_res/stage1_whatif_harmful
RUN_DIR=$BASE_OUT/${SLURM_JOB_ID}
TIMING_WARMUP_ITERS=${WHATIF_TIMING_WARMUP_ITERS:-20}
TIMING_MEASURE_ITERS=${WHATIF_TIMING_MEASURE_ITERS:-200}
NSYS_WARMUP_ITERS=${WHATIF_NSYS_WARMUP_ITERS:-3}
NSYS_MEASURE_ITERS=${WHATIF_NSYS_MEASURE_ITERS:-5}
RUN_NSYS=${WHATIF_RUN_NSYS:-1}
INCLUDE_FUSED=${WHATIF_INCLUDE_FUSED:-0}
VARIANT_FILTER=${WHATIF_VARIANTS:-}

mkdir -p "$RUN_DIR" "$RUN_DIR/nsys"
cd "$MEGATRON_LM_DIR"

export PYTHONPATH="$MEGATRON_LM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-128}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}

echo "=== fixed-model harmful-overlap what-if ==="
echo "host=$(hostname)"
echo "run_dir=$RUN_DIR"
echo "env=$ENV_FILE"
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
echo "timing_iters=$TIMING_MEASURE_ITERS warmup=$TIMING_WARMUP_ITERS"
echo "nsys_iters=$NSYS_MEASURE_ITERS warmup=$NSYS_WARMUP_ITERS run_nsys=$RUN_NSYS"
echo "include_fused=$INCLUDE_FUSED variants=${VARIANT_FILTER:-all}"

GEN_ARGS=(
  --out-dir "$RUN_DIR"
  --timing-warmup-iters "$TIMING_WARMUP_ITERS"
  --timing-measure-iters "$TIMING_MEASURE_ITERS"
  --nsys-warmup-iters "$NSYS_WARMUP_ITERS"
  --nsys-measure-iters "$NSYS_MEASURE_ITERS"
)
if [[ "$INCLUDE_FUSED" == "1" ]]; then
  GEN_ARGS+=(--include-fused)
fi
if [[ -n "$VARIANT_FILTER" ]]; then
  GEN_ARGS+=(--variants "$VARIANT_FILTER")
fi

python3 "$BASE_OUT/generate_whatif_cases.py" "${GEN_ARGS[@]}"

echo "=== timing variants ==="
while IFS=$'\t' read -r VARIANT MANIFEST NSYS_CASE NCCL_ALGO_VALUE NCCL_PROTO_VALUE NCCL_MIN_CHANNELS; do
  echo "--- timing variant=$VARIANT profile=$NCCL_ALGO_VALUE/$NCCL_PROTO_VALUE min_channels=${NCCL_MIN_CHANNELS:-unset} ---"
  if [[ -n "${NCCL_MIN_CHANNELS:-}" ]]; then
    CHANNEL_CMD="export NCCL_MIN_NCHANNELS=$NCCL_MIN_CHANNELS"
  else
    CHANNEL_CMD="unset NCCL_MIN_NCHANNELS"
  fi
  if ! srun -u \
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
      export NCCL_ALGO=$NCCL_ALGO_VALUE
      export NCCL_PROTO=$NCCL_PROTO_VALUE
      $CHANNEL_CMD
      echo \"variant=$VARIANT CUDA_DEVICE_MAX_CONNECTIONS=\$CUDA_DEVICE_MAX_CONNECTIONS NCCL_ALGO=\${NCCL_ALGO:-unset} NCCL_PROTO=\${NCCL_PROTO:-unset} NCCL_MIN_NCHANNELS=\${NCCL_MIN_NCHANNELS:-unset}\"
      torchrun --standalone --nproc_per_node=4 -m kernel_simulator \
        --stage1-run-manifest $MANIFEST
    "
  then
    echo "variant=$VARIANT failed; continuing" | tee -a "$RUN_DIR/failed_variants.txt"
  fi
done < "$RUN_DIR/run_list.tsv"

echo "=== aggregate timing rows ==="
python3 -m kernel_simulator --stage1-aggregate "$RUN_DIR/raw" || true

if [[ "$RUN_NSYS" == "1" ]]; then
  echo "=== select nsys candidate ==="
  python3 "$BASE_OUT/select_nsys_case.py" \
    --atlas-jsonl "$RUN_DIR/atlas.jsonl" \
    --variants-json "$RUN_DIR/variants.json" \
    --output-env "$RUN_DIR/selected_nsys_env.sh" \
    --output-json "$RUN_DIR/selected_nsys_case.json"
  # shellcheck disable=SC1090
  source "$RUN_DIR/selected_nsys_env.sh"
  if [[ -n "$WHATIF_SELECTED_NCCL_MIN_NCHANNELS" ]]; then
    SELECTED_CHANNEL_CMD="export NCCL_MIN_NCHANNELS=$WHATIF_SELECTED_NCCL_MIN_NCHANNELS"
  else
    SELECTED_CHANNEL_CMD="unset NCCL_MIN_NCHANNELS"
  fi
  echo "=== nsys sample variant=$WHATIF_SELECTED_VARIANT reason=$WHATIF_SELECTED_REASON ==="
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
      export NCCL_ALGO=$WHATIF_SELECTED_NCCL_ALGO
      export NCCL_PROTO=$WHATIF_SELECTED_NCCL_PROTO
      $SELECTED_CHANNEL_CMD
      if command -v nsys >/dev/null 2>&1; then
        echo \"nsys variant=$WHATIF_SELECTED_VARIANT case=$WHATIF_SELECTED_NSYS_CASE NCCL_MIN_NCHANNELS=\${NCCL_MIN_NCHANNELS:-unset}\"
        torchrun --standalone --nproc_per_node=4 --no-python \
          nsys profile \
            -t cuda,nvtx,cublas,cudnn \
            --sample=none \
            --cpuctxsw=none \
            --force-overwrite=true \
            -o $RUN_DIR/nsys/${WHATIF_SELECTED_VARIANT}_rank%q{RANK} \
            python3 -m kernel_simulator \
              --case $WHATIF_SELECTED_NSYS_CASE \
              --mode offset_sweep
      else
        echo 'nsys not found inside the job environment; timing results are still available.'
      fi
    "
else
  echo "=== nsys disabled by WHATIF_RUN_NSYS=$RUN_NSYS ==="
fi

echo "=== result files ==="
find "$RUN_DIR" -maxdepth 4 -type f | sort
