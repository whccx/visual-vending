"""容器内 HOI-DETR 批量工作进程：输出识别结果、检测框和最终轨迹。"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2

from app import config
from app.recognition.yoloe_visual_worker import emit


def build_hoi_visualization(candidates_by_frame, hands_by_frame, primary):
    """保留同帧多个商品候选框和所有手框，轨迹帧号从 1 开始。"""
    if len(candidates_by_frame) != len(hands_by_frame):
        raise ValueError('商品框和手框的帧数不一致')
    return {
        'boxes': [None] * len(candidates_by_frame),
        'candidate_boxes': [
            [[float(v) for v in candidate['box']] for candidate in frame]
            for frame in candidates_by_frame
        ],
        'hands': [
            [[float(v) for v in hand['box']] for hand in frame]
            for frame in hands_by_frame
        ],
        'track': [
            {
                'frame': int(point['frame']),
                'box': [float(v) for v in point['box']],
                'point': [float(v) for v in point['point']],
            }
            for point in (primary or {}).get('points', [])
        ],
    }


def recognize_video(video, cabinet_id, dino, state, gate):
    """按照 /recognize-hoi-detr 的检测、过滤、轨迹和评分流程处理视频。"""
    from app.recognition.hoi_mask_matching import rank_masked_track
    from app.catalog.product_catalog import require_cabinet
    from app.detection import area_detection
    from app.detection.hand_area_gate import filter_hand_linked_candidates
    from app.recognition.product_matching import accept_ranking
    from app.tracking.trajectory_hoi_detr import select_visually_moving_track

    require_cabinet(cabinet_id)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened() or not cap.read()[0]:
            raise ValueError('无法读取视频或视频为空')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    flags = gate.scan(str(video)) if gate else None
    config.JOBS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='job_hoi_visual_', dir=config.JOBS) as temp:
        job = Path(temp)
        output = job / 'detections.json'
        command = [str(config.HOI_PYTHON), '-m', 'app.workers.sku_hoi_detr_worker', str(video), str(output)]
        if flags is not None:
            person_file = job / 'person_flags.json'
            person_file.write_text(json.dumps(flags), encoding='utf-8')
            command.extend(['--person-flags', str(person_file)])
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH'):
            env.pop(key, None)
        env['OMP_NUM_THREADS'] = '4'
        with (job / 'worker.log').open('w') as log:
            worker = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900, cwd=str(config.ROOT), env=env)
        if worker.returncode:
            detail = (job / 'worker.log').read_text(errors='replace')[-4000:]
            raise RuntimeError(f'HOI-DETR 检测失败：{detail}')
        detection = json.loads(output.read_text(encoding='utf-8'))
    areas = area_detection.get_area_boxes(video, state)
    frames, _ = filter_hand_linked_candidates(detection['hands'], detection['frames'], areas)
    primary, _ = select_visually_moving_track(video, frames)
    if primary:
        # 与 HOI 接口一致：靠近画面边缘的残片不参与匹配。
        mx, my = 0.01 * width, 0.01 * height
        primary = dict(primary, points=[
            point for point in primary['points']
            if point['box'][0] >= mx and point['box'][1] >= my
            and point['box'][2] <= width - mx and point['box'][3] <= height - my
        ])
    ranking, count, _ = rank_masked_track(video, primary, dino, cabinet_id, state)
    best = accept_ranking(ranking, count)
    return {
        'ok': True,
        'frames': len(frames),
        'track_points': count,
        'product': f"product_{best['sku']}" if best else None,
        'visualization': build_hoi_visualization(frames, detection['hands'], primary),
    }


def main():
    """一次加载商品模型，逐行处理视频；HOI 检测使用独立运行环境。"""
    import torch
    from transformers import AutoModel

    from app.detection.person_gate_client import PersonGateClient

    dino = AutoModel.from_pretrained(str(config.MODEL), local_files_only=True).eval().cuda()
    state = SimpleNamespace()
    gate = None
    try:
        try:
            gate = PersonGateClient()
        except Exception as exc:
            print(f'人物门控不可用，回退到逐帧检测：{exc}', file=sys.stderr, flush=True)
        emit({'ready': True})
        for line in sys.stdin:
            try:
                task = json.loads(line)
                relative = Path(task['video'])
                video = (config.ROOT / relative).resolve()
                if relative.is_absolute() or not video.is_relative_to(config.ROOT.resolve()):
                    raise ValueError('视频必须位于项目目录内')
                if not video.is_file():
                    raise FileNotFoundError(f'视频已被删除或移动：{video}')
                if gate is None or gate.disabled:
                    if gate:
                        gate.close()
                    try:
                        gate = PersonGateClient()
                    except Exception as exc:
                        gate = None
                        print(f'人物门控无法重启：{exc}', file=sys.stderr, flush=True)
                emit(recognize_video(video, task['cabinet_id'], dino, state, gate))
            except Exception as exc:
                emit({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    main()
