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
    if upload is None:
        return "请选择 JPG/PNG 图像。"
    suffix = Path(upload.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return "仅支持 JPG/PNG 图像。"
    if upload.size and upload.size > 8 * 1024 * 1024:
        return "图像大小不能超过 8 MB。"
    return None


def index(request: HttpRequest) -> HttpResponse:
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
    with destination.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    try:
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
    return render(request, "predictor/history.html", {"records": Prediction.objects.all()[:20]})
