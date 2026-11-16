"""Local-only multimodal search; never falls back to a paid/model API."""
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import threading
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError, Image
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/visual", tags=["local-visual-search"])
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class VisualSearch:
    def __init__(self, directory, device):
        from app.visual.index import ProductIndex
        from app.visual.encoder import ClipEncoder
        self.index = ProductIndex(directory)
        config = self.index.metadata["encoder"]
        adapter = Path(directory) / "adapter.npz" if config["adapter_sha256"] else None
        self.encoder = ClipEncoder(config["model"], config["pretrained"], device, adapter)
        if self.encoder.identity() != config:
            raise ValueError("Encoder and FAISS snapshot mismatch")
        self.lock = threading.Lock()

    def search(self, image, text, **filters):
        from app.visual.encoder import normalize
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "检索繁忙，请稍后重试。")
        try:
            started = time.perf_counter()
            vector = self.encoder.images([image]) if image is not None else None
            if text:
                text_vector = self.encoder.texts([text])
                vector = text_vector if vector is None else normalize(.85 * vector + .15 * text_vector)
            results = self.index.search(vector, **filters)
            return {"results": results, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "encoder": self.encoder.identity(), "candidate_only": True,
                    "notice": "相似度用于候选排序，不代表同款概率；请核对型号和商品来源。价格为导入资料中的价格。",
                    "text_note": "默认模型以英文文本为主，中文查询效果需要单独评估。" if text else None}
        finally:
            self.lock.release()


def get_service(request):
    service = getattr(request.app.state, "visual_search", None)
    if service is None:
        raise HTTPException(503, "本地商品索引尚未就绪，请先导入商品图片并构建索引。")
    return service


@router.get("/status")
def status(request: Request):
    service = getattr(request.app.state, "visual_search", None)
    return {"ready": service is not None,
            "products": service.index.metadata["products"] if service else 0,
            "encoder": service.encoder.identity() if service else None}


@router.post("/search")
async def search(request: Request, image: UploadFile | None = File(default=None),
                 text: str = Form(default="", max_length=512),
                 category: str | None = Form(default=None, max_length=100),
                 brand: str | None = Form(default=None, max_length=100),
                 max_price: float | None = Form(default=None, ge=0),
                 attributes: str = Form(default="{}", max_length=2000),
                 prefer_low_price: bool = Form(default=False), limit: int = Form(default=5, ge=1, le=20)):
    if not image and not text.strip():
        raise HTTPException(422, "请上传图片或输入商品描述。")
    if max_price is not None and not math.isfinite(max_price):
        raise HTTPException(422, "预算必须是有限数值。")
    try:
        attrs = json.loads(attributes)
        if not isinstance(attrs, dict) or len(attrs) > 20 or not all(isinstance(k, str) and isinstance(v, str) for k, v in attrs.items()):
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(422, "属性应为字符串键值对，例如 {\"color\":\"black\"}。") from None
    decoded = None
    if image:
        try:
            raw = await image.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "图片不能超过 8MB。")
            from app.visual.data import read_image
            decoded = await run_in_threadpool(read_image, BytesIO(raw))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "图片无效或超过 2000 万像素，请换一张照片。") from None
        finally:
            await image.close()
    service = get_service(request)
    try:
        return await run_in_threadpool(service.search, decoded, text.strip(), limit=limit,
                                      category=category, brand=brand, max_price=max_price,
                                      attributes=attrs, prefer_low_price=prefer_low_price)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("visual_search_failed type=%s", type(exc).__name__)
        raise HTTPException(503, "本地检索暂时失败，请稍后重试。") from None


@router.get("/images/{image_id}")
def gallery_image(image_id: str, request: Request):
    service = get_service(request)
    if image_id not in {row["image_id"] for row in service.index.rows}:
        raise HTTPException(404, "图片不存在。")
    return FileResponse(service.index.directory / "images" / f"{image_id}.jpg", media_type="image/jpeg")
