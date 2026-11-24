"""与现有检测器运行环境隔离的 HOI-DETR 视频识别接口。"""

import logging
import tempfile
import threading
import time
from pathlib import Path

import cv2
import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import Field

from app import config
from app.api.cabinet_validation import require_known_cabinet
from app.api.recognition_common import RecognitionResponse, save_upload
from app.detection import camera_block
from app.detection.hand_area_gate import cabinet_masks_for_candidates, filter_hand_linked_candidates
from app.detection.hoi_tensorrt_client import (
    HOIDetectorError,
    HOIDetectorTimeout,
    detect_video,
)
from app.detection.product_taken import (
    _detect_hands,
    _detect_products,
    _fill_missing_areas,
    _near_hand,
    product_departure_evidence,
)
from app.recognition.hoi_mask_matching import rank_masked_track
from app.recognition.product_matching import accept_ranking
from app.tracking.trajectory_hoi_detr import select_visually_moving_track

ROOT = config.ROOT
JOBS = config.JOBS
RESOURCE_LOCK = threading.Lock()
LOG = logging.getLogger("sku." + __name__)
router = APIRouter()


class ResponseHOI(RecognitionResponse):
    """HOI 识别结果及可选的掩码检查目录。"""

    status: str = Field(
        description=(
            "ok=识别到已登记商品；unknown_product=有商品被拿走但未识别出具体商品；"
            "door_not_opened=未开柜门；door_opened_no_taken=开门未拿商品；"
            "camera_blocked=疑似摄像头被遮挡（开门后连续黑帧无货柜，画面恢复开局"
            "或结尾黑得异于关门状态）；"
            "not_found=拿走检测失败时的兜底"
        )
    )
    debug_dir: str | None = None


def _judge_frames(detection, masks=None):
    """把 HOI 逐帧检测结果整理成拿走判定所需的帧结构。

    只有原始检出货柜的帧，其商品才参与判定，且商品必须与手关联、面积不超过
    手掌倍数上限：HOI 会把人体误检为商品并关联到手，这类框会跟随手离开
    货架，伪造持物离架事件；货柜漏检帧的中位补全仅用于距离度量。
    """
    ratio = config.hoi_max_product_hand_area_ratio()

    def box_area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    frames = []
    if masks is None:
        masks = [None] * len(detection["areas"])
    for area, hands, objects, mask in zip(
        detection["areas"], detection["hands"], detection["frames"], masks
    ):
        products = []
        if area is not None:
            for obj in objects:
                linked = [hands[i] for i in obj["hand_indices"] if i < len(hands)]
                if not linked:
                    continue
                hand_area = max(box_area(hand["box"]) for hand in linked)
                if box_area(obj["box"]) <= ratio * hand_area:
                    products.append(obj["box"])
        frames.append({"area": area, "area_mask": mask, "hands": hands, "products": products})
    return frames


def _supplement_frames(video_path, raw_frames, frames, state):
    """HOI 证据不足时补充区域掩码、MediaPipe 手部与持物商品检测。

    只扫描首末货柜检出帧之间的范围：缺掩码帧用区域 seg 模型补柜内掩码
    （拿走判定以掩码为区域参考），无手帧用 MediaPipe 四方向检测兜底
    （优先 GPU，回退 CPU），无持物商品帧用 held product 模型检测并保留
    手旁候选。直接就地更新 frames；返回补充统计。

    参数：
        video_path: 视频路径，用于逐帧解码。
        raw_frames: 补全前的判定帧，用于确定货柜检出范围。
        frames: 中位补全后的判定帧，就地补充掩码、手与商品。
        state: 当前请求应用状态，提供共享模型。
    返回：
        (补充的手数, 补充的持物商品数, 补充的掩码数, 扫描帧数)。
    """
    from app.detection import area_detection

    indices = [i for i, frame in enumerate(raw_frames) if frame.get("area") is not None]
    if not indices:
        return 0, 0, 0, 0
    gpu_pipeline = getattr(state, "gpu_pipeline", None)
    gpu_hands = gpu_pipeline.hands if gpu_pipeline is not None else None
    added_hands = added_products = added_masks = scanned = 0
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in range(indices[0], indices[-1] + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                LOG.warning("拿取证据兜底在第 %s 帧解码失败，提前结束", index)
                break
            frame = frames[index]
            if frame.get("area_mask") is None:
                scanned += 1
                if not hasattr(state, "area_model"):
                    state.area_model = area_detection.load_model()
                mask, mask_box = area_detection.detect_frame_mask(image, state.area_model)
                if mask is not None:
                    frame["area_mask"] = mask
                    if frame.get("area") is None:
                        frame["area"] = mask_box
                    added_masks += 1
            if not frame["hands"]:
                scanned += 1
                frame["hands"] = _detect_hands(image, state.hands, gpu_hands)
                added_hands += len(frame["hands"])
            # 手旁持物商品是拿走判定的必要证据；HOI 商品被面积上限丢弃时由此找回。
            if not frame["products"] and frame["hands"]:
                scanned += 1
                candidates = _detect_products(image, state.detector)
                kept = [
                    box for box in candidates
                    if any(_near_hand(box, hand["box"]) for hand in frame["hands"])
                ]
                if kept:
                    frame["products"] = kept
                    added_products += len(kept)
    finally:
        capture.release()
    return added_hands, added_products, added_masks, scanned


def _classify_without_match(detection, video_path, state, masks=None):
    """未识别出商品时细分状态：未知商品 / 未开柜门 / 开门未拿商品。

    直接复用 HOI TensorRT 的逐帧货柜框、手框和商品框喂给拿走判定，
    不再对整段视频做二次推理；判定异常时回退 not_found。拿取动作快、
    遮挡或长袖只露指尖常令 HOI 手与持物商品证据断档，未判出拿走时再以
    MediaPipe 手部与 held product 持物检测兜底补充一次。
    """
    area_frames = sum(area is not None for area in detection["areas"])
    try:
        raw_frames = _judge_frames(detection, masks)
        frames = _fill_missing_areas([dict(frame) for frame in raw_frames])
        departure = product_departure_evidence(frames)
        if departure is None and area_frames >= config.hoi_door_min_area_frames():
            try:
                frames = [dict(frame) for frame in raw_frames]
                added_hands, added_products, added_masks, scanned = _supplement_frames(
                    video_path, raw_frames, frames, state
                )
            except Exception:
                LOG.exception("HOI 拿取证据兜底失败，按原始证据判定")
                added_hands = added_products = added_masks = scanned = 0
            if scanned:
                frames = _fill_missing_areas(frames)
                LOG.info(
                    "HOI 拿取证据兜底：扫描 %s 帧，补充 MediaPipe 手 %s 个、"
                    "手旁持物商品 %s 个、区域掩码 %s 帧",
                    scanned, added_hands, added_products, added_masks,
                )
                departure = product_departure_evidence(frames)
    except Exception:
        LOG.exception("HOI-DETR 拿走判定失败，状态回退为 not_found")
        return "not_found", area_frames, None
    if departure is not None:
        return "unknown_product", area_frames, departure
    if area_frames < config.hoi_door_min_area_frames():
        return "door_not_opened", area_frames, None
    return "door_opened_no_taken", area_frames, None


@router.post(
    "/recognize-hoi-detr",
    response_model=ResponseHOI,
    tags=["商品识别"],
    summary="HOI-DETR GPU 视频商品识别",
)
def recognize_hoi_detr(
    request: Request, cabinet_id: str = Form(..., description="柜子编码"), video: UploadFile = File(..., description="待识别的视频文件"),
    save_debug: bool = Form(False, description="保存最终轨迹的商品裁剪图和掩码供检查"),
):
    """调用独立 TensorRT 服务检测人手和商品，再对运动轨迹进行商品匹配。

    未识别出商品时用 HOI 逐帧检测框直接做拿走判定并细分状态：有商品被
    拿走返回 unknown_product，否则按货柜检出帧数区分 door_not_opened
    （低于阈值）与 door_opened_no_taken。返回前检查摄像头遮挡：确认过
    货柜检出后连续黑帧无货柜，且画面恢复开局画面、或黑帧持续到结尾且
    结尾比开局关门黑段还黑时，状态改为 camera_blocked，其余字段不变。
    参数：
        request: 当前请求，提供共享模型和区域检测状态。
        video: 上传的视频文件。
    返回：
        包含商品候选、分数、轨迹点数和耗时的 RecognitionResponse。
    异常：
        输入无效时返回 HTTP 400，检测超时返回 HTTP 504，其他识别错误返回 HTTP 500。
    """
    start = time.monotonic()
    original = (video.filename or "").replace("\\", "/").split("/")[-1]
    if (
        not original
        or original in (".", "..")
        or any(ord(c) < 32 for c in original)
        or not Path(original).suffix
        or not Path(original).stem
    ):
        video.file.close()
        raise HTTPException(status_code=400, detail="请提供有效的视频文件名（包含扩展名）")
    require_known_cabinet(cabinet_id, video)
    JOBS.mkdir(parents=True, exist_ok=True)
    with RESOURCE_LOCK, tempfile.TemporaryDirectory(prefix="job_hoi_detr_", dir=JOBS) as tmp:
        try:
            job = Path(tmp)
            video_path = save_upload(video, original, job, cabinet_id)
            probe = cv2.VideoCapture(str(video_path))
            try:
                if not probe.isOpened() or not probe.read()[0]:
                    raise ValueError("无法读取视频或视频为空")
                width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                probe.release()
            from app.detection import area_detection
            if not hasattr(request.app.state, "area_model"):
                request.app.state.area_model = area_detection.load_model()
            areas, packed_masks, shape = area_detection.detect_video_masks(
                video_path, request.app.state.area_model
            )
            detection = detect_video(video_path, frame_count, areas)
            LOG.info("HOI TensorRT 检测耗时：%s", detection.get("timings"))
            masks = cabinet_masks_for_candidates(
                packed_masks, shape, detection["hands"], detection["frames"]
            )
            frames, hand_area = filter_hand_linked_candidates(
                detection["hands"],
                detection["frames"],
                detection["areas"],
                max_linked_hand_area_ratio=config.hoi_max_product_hand_area_ratio(),
                require_departure=True,
                hand_track_base_distance=config.hoi_hand_track_base_distance(),
                hand_track_iou_threshold=config.hoi_hand_track_iou_threshold(),
                area_masks_by_frame=masks,
            )
            LOG.info("HOI-DETR 手部区域筛选统计：%s", hand_area)
            primary, motion_summary = select_visually_moving_track(video_path, frames)
            LOG.info("HOI-DETR 运动验证统计：%s", motion_summary)
            if primary:
                # 靠近画面边缘的框可能只包含商品残片；保留轨迹，但不用于商品匹配。
                mx, my = 0.01 * width, 0.01 * height
                rank_points = [
                    point
                    for point in primary["points"]
                    if point["box"][0] >= mx
                    and point["box"][1] >= my
                    and point["box"][2] <= width - mx
                    and point["box"][3] <= height - my
                ]
                primary = dict(primary, points=rank_points)
                LOG.info("HOI-DETR 参与排序帧数：%s", len(rank_points))
            ranking, count, debug_dir = rank_masked_track(
                video_path, primary, request.app.state.dino, cabinet_id,
                request.app.state, save_debug=save_debug, use_trt_bg0=True,
            )
            best = accept_ranking(ranking, count)
            status = "ok"
            if not best:
                status, area_frames, _ = _classify_without_match(
                    detection, video_path, request.app.state, masks
                )
                LOG.info(
                    "HOI-DETR 未识别出商品，细分状态：%s（货柜检出帧数=%s）", status, area_frames
                )
            if camera_block.judge_camera_blocked(video_path, detection["areas"]):
                LOG.info("HOI-DETR 判定摄像头被遮挡，状态由 %s 改为 camera_blocked", status)
                status = "camera_blocked"
            return ResponseHOI(
                cabinet_id=cabinet_id,
                status=status,
                product=f"product_{best['sku']}" if best else None,
                confidence=round(best["combined"], 6) if best else None,
                vector_score=round(best["vector"], 6) if best else None,
                color_score=round(best["color"], 6) if best else None,
                track_points=count,
                frames=len(frames),
                video_path=str(video_path.relative_to(ROOT)),
                duration=round(time.monotonic() - start, 3),
                debug_dir=str(debug_dir.relative_to(ROOT)) if debug_dir else None,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except HOIDetectorTimeout as exc:
            LOG.exception("HOI TensorRT 服务检测超时")
            raise HTTPException(status_code=504, detail="HOI TensorRT 服务检测超时") from exc
        except HOIDetectorError as exc:
            LOG.exception("HOI TensorRT 服务调用失败")
            raise HTTPException(status_code=502, detail="HOI TensorRT 服务不可用或返回异常") from exc
        except Exception:
            LOG.exception("HOI-DETR 识别失败")
            raise HTTPException(status_code=500, detail="HOI-DETR 识别失败，请查看容器日志")
        finally:
            video.file.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
