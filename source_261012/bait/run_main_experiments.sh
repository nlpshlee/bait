#!/usr/bin/env bash
# Usage: bash run_main_experiments.sh [model] [gpu_a] [gpu_b] [retain] [renorm|original]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

model_name="${1:-Llama-3.2-3B}"
gpu_a="${2:-2}"
gpu_b="${3:-3}"
n=1000
selection_run="dummy_intervention"
selection_n=1000
default_retain="${4:-0.75}"  # 유지율 0.75 = 억제 25%. 기존 전체 평균 충실도가 가장 높은 설정.
attention_mode="${5:-renorm}"
case "$attention_mode" in
  renorm|original) ;;
  *) echo "다섯 번째 인자는 renorm 또는 original이어야 합니다." >&2; exit 2 ;;
esac

# 각 행은 전체 fact/counter를 평가하는 독립 실행. 모든 행을 동시에 실행한다.
# GPU  방법     MLP층  MHA층  층당헤드  잔차층  유지율
# 행마다 $default_retain 대신 1, 0.75, 0.5, 0.25, 0 등을 지정해도 된다.
experiments=(
  # 기존 실험: 완료된 결과를 비교에 재사용한다.
  # "$gpu_a vanilla 0 0 0 0 0.5"
  # "$gpu_a dummy   0 0 0 0 0.5"
  # "$gpu_b ours    1 4 1 1 0.5"
  # "$gpu_b ours    1 1 4 1 0.5"
  # "$gpu_a ours    3 3 4 3 0.5"
  # "$gpu_a ours    5 5 4 5 0.5"
  # "$gpu_b ours    5 5 8 5 0.5"
  # "$gpu_b ours    10 10 8 10 0.5"
  # "$gpu_a ours    5 5 8 5 0.75"
  # "$gpu_a ours    5 5 8 5 0.25"
  # "$gpu_b ours    5 5 8 5 0.0"

  # 기존 모듈 제거 실험: 결과를 비교에 재사용한다.
  # "$gpu_a ours 0 5 8 5 $default_retain"  # MLP 제외
  # "$gpu_a ours 5 0 0 5 $default_retain"  # MHA 제외
  # "$gpu_b ours 5 5 8 0 $default_retain"  # 잔차 제외
  # "$gpu_b ours 0 5 8 0 $default_retain"  # MHA만 사용

  # 동일한 저장 후보·유지율로 MHA 재정규화 유무만 비교한다.
  "$gpu_a ours 5 5 8 5 $default_retain"  # 전체 모듈, 기본 GPU 2
  "$gpu_b ours 0 5 8 5 $default_retain"  # MLP 제외, 기본 GPU 3
)

log_dir="../../outputs/main_experiments/$model_name/logs"
mkdir -p "$log_dir"
for experiment in "${experiments[@]}"; do
  read -r gpu method mlp_layers mha_layers heads_per_layer residual_layers retain <<< "$experiment"
  name="$method"
  attention_args=()
  if [[ "$method" == ours ]]; then
    LC_NUMERIC=C printf -v keep '%g' "$retain"
    name+="_m${mlp_layers}_a${mha_layers}_h${heads_per_layer}_r${residual_layers}_keep${keep}"
    if [[ "$attention_mode" == renorm ]]; then
      attention_args+=(--attention-renorm)
      name+="_renorm"
    fi
  fi
  name+="_n${n}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -B -u main_experiments.py \
    --model "$model_name" --method "$method" --gpu "$gpu" --n "$n" \
    --mlp-layers "$mlp_layers" --mha-layers "$mha_layers" \
    --heads-per-layer "$heads_per_layer" --residual-layers "$residual_layers" \
    --retain "$retain" --selection-run "$selection_run" --selection-n "$selection_n" \
    "${attention_args[@]}" \
    --run-name "$name" > "$log_dir/$name.log" 2>&1 &
  echo "method=$method GPU=$gpu MLP=$mlp_layers MHA=$mha_layers×$heads_per_layer residual=$residual_layers retain=$retain attention=$attention_mode PID=$! log=$log_dir/$name.log"
done
