#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ $# -ne 1 || ( "$mode" != "1" && "$mode" != "2" && "$mode" != "3" ) ]]; then
  printf '사용법: bash %s {1|2|3}\n  1: 4개 shard 측정 시작\n  2: 완료된 결과 분석\n  3: 추론 중 현재까지의 결과로 중간 분석\n' "$0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
cd -- "$script_dir"

python_bin="${BAIT_PYTHON:-/home/nlpshlee/.conda/envs/bait/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Python 실행 파일을 찾을 수 없습니다: %s\nBAIT_PYTHON에 bait 환경의 Python 경로를 지정하세요.\n' "$python_bin" >&2
  exit 1
fi

model_name="Llama-3.2-3B"
run_name="${RUN_NAME:-main1000_last_balanced_with44}"
n=1000  # 분할별 1,000문항: fact / counter 합계 2,000문항
splits=(fact counter)
# 기존 9개 문서의 10개 비율에, 사실/반사실 수가 같은 4:4(8개 문서)를 추가한다.
ratios=(0:9 1:8 2:7 3:6 4:5 4:4 5:4 6:3 7:2 8:1 9:0)
gpu_ids=(2 3)
shards_per_gpu=2
num_shards=$((${#gpu_ids[@]} * shards_per_gpu))

run_dir="$project_dir/outputs/check_confirmation_bias/$model_name/$run_name"
mkdir -p "$run_dir/logs"

case "$mode" in
  1)
    for ((shard = 0; shard < num_shards; shard++)); do
      gpu="${gpu_ids[$((shard / shards_per_gpu))]}"
      log="$run_dir/logs/shard${shard}.log"
      nohup "$python_bin" -B -u check_confirmation_bias.py \
        --model "$model_name" \
        --n "$n" \
        --splits "${splits[@]}" \
        --ratios "${ratios[@]}" \
        --positions last \
        --gpu "$gpu" \
        --shard "$shard" \
        --num-shards "$num_shards" \
        --run-name "$run_name" \
        >> "$log" 2>&1 < /dev/null &
      printf 'shard=%s GPU=%s PID=%s 로그=%s\n' "$shard" "$gpu" "$!" "$log"
    done
    printf '측정을 백그라운드에서 시작했습니다. 네 shard가 정상 완료된 뒤 인자 2로 분석하세요.\n'
    ;;
  2|3)
    analysis_args=(--analyze-only)
    analysis_log="$run_dir/logs/analyze.log"
    if [[ "$mode" == "3" ]]; then
      analysis_args+=(--preview)
      analysis_log="$run_dir/logs/analyze_preview.log"
    fi
    printf '분석 로그: %s\n' "$analysis_log"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python_bin" -B -u check_confirmation_bias.py \
      --model "$model_name" \
      --run-name "$run_name" \
      --splits "${splits[@]}" \
      "${analysis_args[@]}" \
      >> "$analysis_log" 2>&1
    printf '분석 완료: %s\n' "$run_dir"
    ;;
esac
