"""检测请求开始前校验柜子。"""
from fastapi import HTTPException
from app.catalog import product_catalog


def require_known_cabinet(cabinet_id, video=None):
    try:
        product_catalog.require_cabinet(cabinet_id)
    except product_catalog.CabinetNotFound as exc:
        if video is not None:
            video.file.close()
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        if video is not None:
            video.file.close()
        raise HTTPException(status_code=400, detail=str(exc))
