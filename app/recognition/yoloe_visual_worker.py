"""容器内批量 YOLOE 可视化工作进程，复用视频识别服务的模型和判定逻辑。"""

import json
import sys
from pathlib import Path

from app import config

PREFIX = 'YOLOE_VISUAL_RESULT '


def emit(data):
    """使用独立前缀区分 JSON 响应与第三方库控制台日志。"""
    print(PREFIX + json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    """一次加载模型，按标准输入逐行读取视频任务。"""
    from transformers import AutoModel

    from app.catalog.product_catalog import require_cabinet
    from app.detection.area_detection import load_model
    from app.detection.held_product_detect import load_detector, load_gpu_hands
    from app.detection.person_gate_client import PersonGateClient
    from app.recognition.gpu_video_pipeline import GPUVideoPipeline
    from app.recognition.product_matching import accept_ranking

    detector, hands = load_detector()
    gpu_hands = None
    gate = None
    try:
        dino = AutoModel.from_pretrained(str(config.MODEL), local_files_only=True).eval().cuda()
        gpu_hands = load_gpu_hands()
        pipeline = GPUVideoPipeline(detector, dino, gpu_hands, batch_size=config.gpu_batch_size())
        area_model = load_model()
        try:
            gate = PersonGateClient()
        except Exception as exc:
            print(f'人物门控不可用，回退到逐帧检测：{exc}', file=sys.stderr, flush=True)
        emit({'ready': True})
        for line in sys.stdin:
            try:
                task = json.loads(line)
                cabinet_id = task['cabinet_id']
                relative = Path(task['video'])
                video = (config.ROOT / relative).resolve()
                if relative.is_absolute() or not video.is_relative_to(config.ROOT.resolve()):
                    raise ValueError('视频必须位于项目目录内')
                if not video.is_file():
                    raise FileNotFoundError(f'视频已被删除或移动：{video}')
                require_cabinet(cabinet_id)
                if gate is None or gate.disabled:
                    if gate:
                        gate.close()
                    try:
                        gate = PersonGateClient()
                    except Exception as exc:
                        gate = None
                        print(f'人物门控无法重启：{exc}', file=sys.stderr, flush=True)
                flags = gate.scan(str(video)) if gate else None
                if not video.is_file():
                    raise FileNotFoundError(f'处理期间视频被删除或移动：{video}')
                result = pipeline.run(
                    video, cabinet_id=cabinet_id, person_flags=flags, area_model=area_model,
                    capture_visualization=True,
                )
                best = accept_ranking(result['ranking'], result['track_points'])
                emit({
                    'ok': True,
                    'frames': result['frames'],
                    'track_points': result['track_points'],
                    'product': f"product_{best['sku']}" if best else None,
                    'visualization': result['visualization'],
                })
            except Exception as exc:
                emit({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if gate:
            gate.close()
        if gpu_hands:
            gpu_hands.close()
        hands.close()


if __name__ == '__main__':
    main()
