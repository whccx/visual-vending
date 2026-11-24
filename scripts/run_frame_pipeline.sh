#!/usr/bin/env bash
# 在两个隔离容器中运行跨视频逐帧实验。
set -euo pipefail

if (( $# < 5 )); then
    printf '用法：%s 输出目录 SKU清单.json 视频1 视频2 视频3 [视频...]\n' "$0" >&2
    exit 2
fi
project_dir=$(cd "$(dirname "$0")/.." && pwd)
output_path=$(realpath -m "$1")
scope_path=$(realpath -e "$2")
shift 2
mkdir -p "$(dirname "$output_path")"
if [[ -e "$output_path" ]]; then
    printf '输出目录已存在：%s\n' "$output_path" >&2
    exit 2
fi
output_parent=$(dirname "$output_path")
container_dir="/bench/$(basename "$output_path")"
run_id="$(date +%H%M%S)-$$"
detector_name="visual-frame-detector-$run_id"
controller_name="visual-frame-controller-$run_id"
video_mounts=()
video_args=()
index=0
for path in "$@"; do
    source_path=$(realpath -e "$path")
    index=$((index + 1))
    container_video="/inputs/$(printf '%02d' "$index")/$(basename "$source_path")"
    video_mounts+=(-v "$source_path:$container_video:ro")
    video_args+=("$container_video")
done
cd "$project_dir"
timeout 30s docker compose run --rm --no-deps -T -v "$output_parent:/bench" \
    worker -m scripts.frame_pipeline init "$container_dir"
start_time=$(python3 -c 'import time; print(time.monotonic())')
timeout 600s docker compose run --rm --no-deps -T --name "$detector_name" \
    --user "$(id -u):$(id -g)" -v "$output_parent:/bench" \
    -v "$project_dir/scripts:/experiment/scripts:ro" -v "$project_dir/app:/experiment/app:ro" \
    -v "$project_dir/vending:/experiment/vending:ro" -e PYTHONPATH=/experiment:/runtime:/service \
    --entrypoint python detector -m scripts.frame_pipeline detector "$container_dir" \
    > "$output_path/detector.log" 2>&1 &
detector_pid=$!
gate_args=()
if [[ ${VENDING_BENCH_GATE:-0} == 1 ]]; then
    gate_args+=(--gate)
fi
cleanup() {
    docker stop -t 3 "$detector_name" >/dev/null 2>&1 || true
    wait "$detector_pid" 2>/dev/null || true
}
trap cleanup EXIT
timeout 600s docker compose run --rm --no-deps -T --name "$controller_name" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display \
    -e YOLO_CONFIG_DIR=/tmp/visual-yolo-config -v "$output_parent:/bench" \
    -v "$scope_path:/input-scope.json:ro" "${video_mounts[@]}" \
    worker -m scripts.frame_pipeline controller "$container_dir" "${video_args[@]}" \
    --sku-scope /input-scope.json "${gate_args[@]}" 2>&1 | tee "$output_path/controller.log"
wait "$detector_pid"
trap - EXIT
python3 - "$output_path/summary.json" "$start_time" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
summary['batch_start_monotonic'] = float(sys.argv[2])
summary['batch_wall_seconds'] = summary['finished_monotonic'] - summary['batch_start_monotonic']
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(f"从启动实验容器到全部结果完成：{summary['batch_wall_seconds']:.2f} 秒")
PY
