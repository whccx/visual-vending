"""应用配置；运行时配置只在调用相应函数时读取。"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("SKU_WORKSPACE", Path(__file__).resolve().parent.parent)).resolve()
LOCAL_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "tests" / "video"
PRODUCT_DIR = ROOT / "tests" / "product"
JOBS = ROOT / ".sku_api_jobs"
HELD_PRODUCT_PATH = Path("models/held_product")
HELD_PRODUCT_DIR = ROOT / HELD_PRODUCT_PATH
MODEL_PATH = HELD_PRODUCT_PATH / "dinov3_vitl"
MODEL = ROOT / MODEL_PATH

# vector_compare 支持在项目目录本地独立执行，保留其原有路径语义。
LOCAL_MODEL = LOCAL_ROOT / MODEL_PATH
AREA_MODEL = ROOT / "models/cabinet_area/best.pt"
RESTORE_DATA_DIR = ROOT / ".sku_restore"
GPU_RUNTIME = ROOT / "runtime/gpu"
DOH_EXPERIMENT = ROOT / "runtime/100doh"
DOH_CHECKPOINT = DOH_EXPERIMENT / "models/handobj_100K_ego.pth"
HOI_REPO = ROOT / "runtime/hoi_detr/HOI-DETR"
HOI_PYTHON = ROOT / "runtime/hoi_detr/env/bin/python"
HOI_MODEL = ROOT / "models/hoi-detr/epoch_5.pth"
PERSON_MODEL = ROOT / "models/yolov8n.pt"
PERSON_WORKER = ROOT / "app/workers/sku_person_gate_scan.py"
PERSON_PYTHON = Path("/opt/conda/bin/python")
HELD_PRODUCT_MODEL = HELD_PRODUCT_DIR / "yoloe-26l-seg.pt"
HAND_MODEL = HELD_PRODUCT_DIR / "hand_landmarker.task"

# BG0 Full 模型及其独立安装的 ONNX Runtime CUDA 环境。
BG0_MODEL = ROOT / "models/bg0/model_fp16.onnx"
BG0_RUNTIME = ROOT / "runtime/bg0_python"
BG0_INPUT_SIZE = 512
BG0_MASK_THRESHOLD = 0.5
LOG_CLEANUP_INTERVAL_SECONDS = 3600
LOG_WARNING_INTERVAL_SECONDS = 60
LOG_LOCK_WAIT_SECONDS = 2

# HOI 掩码质量筛选及相邻帧光流检查（实际分割效果需人工或数据集验证）。
HOI_MASK_MIN_FRACTION = 0.12
HOI_MASK_RELATIVE_FRACTION = 0.95
HOI_MASK_MAX_FRAME_GAP = 5
HOI_MASK_FLOW_MAX_ERROR = 3.0
HOI_MASK_FLOW_MIN_CONSISTENCY = 0.55

# HOI 商品框内抠图可见像素占比不足此值时，不参与商品比对。
HOI_MASK_MATCH_MIN_FRACTION = 0.40

# 商品识别使用连续帧窗口，按窗口内前两名综合分差筛选。
SCORE_WINDOW_FRAMES = 5

# 多个商品的色相分数同时过高时，颜色证据无法可靠区分商品，保守拒识。
SCORE_AMBIGUOUS_HUE_THRESHOLD = 0.98 # color 分达到该值视为"色相高度相似"
SCORE_AMBIGUOUS_HUE_COUNT = 8 # 排序表中 ≥5 个候选都达到该值才判歧义

# 方向识别只作为保守的图像平面旋转依据；无法确认时保持原方向。
HOI_ORIENTATION_ENABLED = False
HOI_ORIENTATION_OCR_DIR = ROOT / "runtime/hoi_orientation_ocr"
HOI_ORIENTATION_OCR_MIN_CONFIDENCE = 0.85
HOI_ORIENTATION_OCR_MIN_CHARS = 4
HOI_ORIENTATION_OCR_MIN_GAP = 0.08
HOI_ORIENTATION_TEXT_ASPECT = 1.5
HOI_ORIENTATION_OCR_TARGET_SIZE = 800

# 去手：用 YOLOE 分割手部并从 BG0 掩码中扣除。
HOI_REMOVE_HAND = True

# 手部分割时扩展商品框，为手指提供相邻画面上下文。
HOI_HAND_CROP_PADDING_RATIO = 0.4

# HOI 商品抠图比对前的饱和度、对比度倍数。
HOI_MATCH_SATURATION = 1.5
HOI_MATCH_CONTRAST = 1.1

# 文件名使用东八区时间；容器系统时间可继续保持 UTC。
LOG_FILENAME_UTC_OFFSET_HOURS = 8


def product_dir() -> Path:
    return Path(os.environ.get("SKU_PRODUCT_DIR", str(PRODUCT_DIR)))


def area_model() -> Path:
    return Path(os.environ.get("SKU_AREA_MODEL", str(AREA_MODEL)))


def area_confidence() -> float:
    """货架区域检测的置信度阈值。"""
    return float(os.environ.get("SKU_AREA_CONF", "0.75"))


def restore_backend() -> str:
    return os.environ.get("SKU_RESTORE_BACKEND", "http://127.0.0.1:8188").rstrip("/")


def restore_data_dir() -> Path:
    return Path(os.environ.get("SKU_RESTORE_DATA_DIR", str(RESTORE_DATA_DIR)))


def min_vector_score() -> float:
    return float(os.environ.get("SKU_MIN_VECTOR_SCORE", "0.20"))


def vector_weight() -> float:
    return float(os.environ.get("SKU_VECTOR_WEIGHT", "0.7"))


def color_weight() -> float:
    return float(os.environ.get("SKU_COLOR_WEIGHT", "0.3"))


def min_color_score() -> float:
    return float(os.environ.get("SKU_MIN_COLOR_SCORE", "0.60"))


def max_product_hand_area_ratio() -> float:
    """YOLOE 最终轨迹中，商品框面积相对于邻近手框面积的最大倍数。"""
    return float(os.environ.get("SKU_MAX_PRODUCT_HAND_AREA_RATIO", "16.0"))


def hoi_max_product_hand_area_ratio() -> float:
    """返回 HOI 商品框相对于关联手掌框的最大面积倍数。"""
    return float(os.environ.get("SKU_HOI_MAX_PRODUCT_HAND_AREA_RATIO", "3.0"))


def hoi_hand_track_base_distance() -> float:
    """HOI 手部轨迹关联中，相邻帧手中心位移的基础距离上限。"""
    return float(os.environ.get("SKU_HOI_HAND_TRACK_BASE_DISTANCE", "90.0"))


def hoi_hand_track_iou_threshold() -> float:
    """HOI 手部轨迹关联中，中心位移超限时的手框 IoU 兜底阈值。"""
    return float(os.environ.get("SKU_HOI_HAND_TRACK_IOU_THRESHOLD", "0.15"))


def hoi_door_min_area_frames() -> int:
    """HOI 未识别商品时，判定已开柜门所需的最少货柜检出帧数。"""
    return int(os.environ.get("SKU_HOI_DOOR_MIN_AREA_FRAMES", "5"))


def hoi_block_confirm_frames() -> int:
    """摄像头遮挡判定中，确认货柜检出或黑帧遮挡状态所需的连续帧数。"""
    return int(os.environ.get("SKU_HOI_BLOCK_CONFIRM_FRAMES", "5"))


def hoi_block_dark_pixel_max() -> int:
    """摄像头遮挡判定中，像素灰度低于该值计为黑像素。"""
    return int(os.environ.get("SKU_HOI_BLOCK_DARK_PIXEL_MAX", "50"))


def hoi_block_dark_frame_ratio() -> float:
    """摄像头遮挡判定中，判定黑帧所需的暗像素占比下限。"""
    return float(os.environ.get("SKU_HOI_BLOCK_DARK_RATIO", "0.40"))


def hoi_block_sample_frames() -> int:
    """摄像头遮挡判定中，开局段与结束段各采样比较的帧数。"""
    return int(os.environ.get("SKU_HOI_BLOCK_SAMPLE_FRAMES", "5"))


def hoi_block_hash_distance() -> int:
    """摄像头遮挡判定中，开局与结束帧 dHash 汉明距离的相似上限。"""
    return int(os.environ.get("SKU_HOI_BLOCK_HASH_DISTANCE", "12"))


def hoi_block_brightness_delta() -> float:
    """摄像头遮挡判定中，开局与结束段平均灰度差的相似上限。"""
    return float(os.environ.get("SKU_HOI_BLOCK_MAX_MEAN_DELTA", "40"))


def hoi_block_tail_ratio() -> float:
    """摄像头遮挡判定中，结尾黑段相对开局黑段平均亮度的上限比例。"""
    return float(os.environ.get("SKU_HOI_BLOCK_TAIL_BRIGHTNESS_RATIO", "0.5"))


def taken_near_area_ratio() -> float:
    """手持商品离架判定中，手靠近货架的对角线比例。"""
    return float(os.environ.get("SKU_TAKEN_NEAR_AREA_RATIO", "0.10"))


def taken_min_distance_gain_ratio() -> float:
    """手持商品离架判定中，相对货架最小外移距离比例。"""
    return float(os.environ.get("SKU_TAKEN_MIN_DISTANCE_GAIN_RATIO", "0.10"))


def taken_min_outward_steps() -> int:
    """手持商品离架判定中，手距离连续增加的最少帧步数。"""
    return int(os.environ.get("SKU_TAKEN_MIN_OUTWARD_STEPS", "2"))


def taken_min_product_frames() -> int:
    """手持商品离架判定中，外移阶段连续持物的最少帧数。"""
    return int(os.environ.get("SKU_TAKEN_MIN_PRODUCT_FRAMES", "2"))


def taken_hand_max_gap() -> int:
    """手持商品离架判定中，同一手轨迹允许的最大漏检帧间隔。"""
    return int(os.environ.get("SKU_TAKEN_HAND_MAX_GAP", "15"))


def taken_mask_near_distance() -> float:
    """掩码区域度量下，手靠近掩码的固定距离（像素）。"""
    return float(os.environ.get("SKU_TAKEN_MASK_NEAR_DISTANCE", "50"))


def taken_mask_gain_distance() -> float:
    """掩码区域度量下，持物离架的最小位移（像素）。"""
    return float(os.environ.get("SKU_TAKEN_MASK_GAIN_DISTANCE", "30"))


def taken_sparse_event_max_gap() -> int:
    """稀疏手部证据中，近货架与离架持物之间允许的最大帧间隔。"""
    return int(os.environ.get("SKU_TAKEN_SPARSE_EVENT_MAX_GAP", "120"))


def frame_workers() -> int:
    return int(os.environ.get("SKU_FRAME_WORKERS", "1"))


def hand_backend() -> str:
    return os.environ.get("SKU_HAND_BACKEND", "cpu")


def gpu_batch_size() -> int:
    return int(os.environ.get("SKU_GPU_BATCH_SIZE", "16"))


def doh_workers() -> int:
    return int(os.environ.get("SKU_100DOH_WORKERS", "4"))


def hoi_detr_batch_size() -> int:
    return int(os.environ.get("SKU_HOI_DETR_BATCH_SIZE", "2"))


def hoi_detr_batch_workers() -> int:
    return int(os.environ.get("SKU_HOI_DETR_BATCH_WORKERS", "1"))


# HOI 常驻池每个进程处理一帧一批，三个 CUDA 推理线程。
HOI_POOL_PROCESS_COUNT = 2
HOI_POOL_THREADS = 3
HOI_POOL_START_TIMEOUT_SECONDS = 180
HOI_POOL_REQUEST_TIMEOUT_SECONDS = 900

# HOI 检测由独立 TensorRT 服务负责；地址可按部署网络覆盖。
HOI_TENSORRT_URL = os.environ.get("SKU_HOI_TENSORRT_URL", "http://detector:8770").rstrip("/")
HOI_TENSORRT_CONNECT_TIMEOUT = 5
HOI_TENSORRT_REQUEST_TIMEOUT = 900


def log_options() -> dict:
    """返回日志选项；无效值由日志初始化层降级处理。"""
    return {
        "directory": Path(os.environ.get("SKU_LOG_DIR", str(LOCAL_ROOT / "logs"))),
        "level": os.environ.get("SKU_LOG_LEVEL", "INFO").upper(),
        "max_bytes": int(os.environ.get("SKU_LOG_MAX_BYTES", str(50 * 1024 * 1024))),
        "backup_count": int(os.environ.get("SKU_LOG_BACKUP_COUNT", "5")),
        "retention_days": int(os.environ.get("SKU_LOG_RETENTION_DAYS", "7")),
        "total_max_bytes": int(os.environ.get("SKU_LOG_TOTAL_MAX_BYTES", str(2 * 1024 ** 3))),
    }
