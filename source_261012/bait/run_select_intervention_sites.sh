#!/usr/bin/env bash
# Usage: bash run_select_intervention_sites.sh 2,3 4 [run_name]

# 실험 설정
model_name="Llama-3.1-8B"
run_name="dummy_intervention"
n=1000
calibration_n=1000
search_ratios=(2:7 4:5 5:4 7:2)

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

gpu_list="${1:?GPU 목록을 지정하세요. 예: 2,3}"
num_shards="${2:?샤드 수를 지정하세요. 예: 4}"
run_name="${3:-$run_name}"
[[ "$gpu_list" =~ ^[0-9]+(,[0-9]+)*$ && "$num_shards" =~ ^[1-9][0-9]*$ ]] || exit 2
IFS=',' read -r -a gpus <<< "$gpu_list"

log_dir="../../outputs/interventions/$model_name/$run_name/logs"
mkdir -p "$log_dir"

for ((shard=0; shard<num_shards; shard++)); do
  gpu="${gpus[$((shard % ${#gpus[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -B -u run_interventions.py calibrate \
    --model "$model_name" --n "$n" --calibration-n "$calibration_n" \
    --ratios "${search_ratios[@]}" \
    --gpu "$gpu" --shard "$shard" --num-shards "$num_shards" \
    --run-name "$run_name" \
    > "$log_dir/calibrate_shard${shard}.log" 2>&1 &
  echo "shard=$shard GPU=$gpu PID=$!"
done
