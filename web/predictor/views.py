"""Django 上传、推理和历史记录视图。

# 【技术栈】Django HttpRequest/HttpResponse + Pillow/PyTorch（通过 src.modeling
# 统一接口）+ SQLite ORM。流程是上传校验 -> 临时保存 -> 模型推理 -> 保存结果 ->
# 渲染结果页；这里不执行训练，也不修改训练 manifest。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

WEB_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEB_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import predict_image  # noqa: E402

from .models import Prediction  # noqa: E402

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _error(request: HttpRequest, message: str) -> HttpResponse:
    return render(request, "predictor/index.html", {"error": message})


def _validate_upload(upload: UploadedFile | None) -> str | None:
    # 【输入校验】限制 JPG/JPEG/PNG 扩展名和 8 MB 大小，减少无效文件进入模型。
    """在写入磁盘和调用模型前校验上传对象。"""
    if upload is None:
        return "请选择 JPG/PNG 图像。"
    suffix = Path(upload.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return "仅支持 JPG/PNG 图像。"
    if upload.size and upload.size > 8 * 1024 * 1024:
        return "图像大小不能超过 8 MB。"
    return None


def index(request: HttpRequest) -> HttpResponse:
    # 【Web 推理流程】Django 接收眼底图像，调用 settings.EYE_DEMO_ARTIFACT 指向的
    # 已训练 artifact，返回 normal/disease 概率、置信度和推理耗时。
    """首页：GET 展示上传表单，POST 保存图像并执行一次预测。"""
    if request.method != "POST":
        return render(request, "predictor/index.html")
    upload = request.FILES.get("image")
    error = _validate_upload(upload)
    if error:
        return _error(request, error)
    assert upload is not None
    upload_dir = Path(settings.EYE_DEMO_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4().hex}{Path(upload.name).suffix.lower()}"
    # 分块写入，避免把上传文件一次性全部放入内存；大小上限由校验控制。
    with destination.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    try:
        # settings.EYE_DEMO_ARTIFACT 默认指向训练好的 ResNet JSON 配置。
        result = predict_image(destination, settings.EYE_DEMO_ARTIFACT)
    except FileNotFoundError:
        return _error(request, "模型权重不存在，请先运行 IDRiD ResNet-18 训练命令。")
    except Exception as exc:
        return _error(request, f"图像处理失败：{exc}")
    record = Prediction.objects.create(
        image_name=upload.name,
        class_name=result["class_name"],
        confidence=result["confidence"],
        probabilities_json=result["probabilities"],
        inference_ms=result["inference_ms"],
    )
    return render(request, "predictor/result.html", {"result": result, "record": record})


def result(request: HttpRequest) -> HttpResponse:
    """展示最近一条预测记录；没有记录时回到上传页面并提示用户。"""
    latest = Prediction.objects.first()
    if latest is None:
        return render(request, "predictor/index.html", {"error": "暂无预测记录，请先上传图像。"})
    result_data = {
        "class_name": latest.class_name,
        "confidence": latest.confidence,
        "probabilities": latest.probabilities_json,
        "inference_ms": latest.inference_ms,
    }
    return render(request, "predictor/result.html", {"result": result_data, "record": latest})


def history(request: HttpRequest) -> HttpResponse:
    """展示最近 20 条预测历史，按模型中的 created_at 倒序排列。"""
    return render(request, "predictor/history.html", {"records": Prediction.objects.all()[:20]})
