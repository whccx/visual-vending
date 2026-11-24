"""视频识别接口共享的响应模型与上传保存工具。"""

from typing import Optional

from pydantic import BaseModel

from app.api.video_storage import store_video
from app.config import VIDEO_DIR


class RecognitionResponse(BaseModel):
    """视频识别接口的响应数据。"""

    cabinet_id: str

    status: str
    product: Optional[str] = None
    confidence: Optional[float] = None
    vector_score: Optional[float] = None
    color_score: Optional[float] = None
    track_points: int = 0
    frames: int
    video_path: str
    duration: float


def save_upload(upload, original, job, cabinet_id):
    """在指定柜子下保存或复用视频。"""
    return store_video(upload, original, job, cabinet_id, VIDEO_DIR)
