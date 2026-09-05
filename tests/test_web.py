"""Web smoke tests for the optional Django demo."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WEB_ROOT = PROJECT_ROOT / "web"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))


django = pytest.importorskip("django")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "eye_demo.settings")
django.setup()


def test_upload_page_contains_medical_disclaimer() -> None:
    from django.test import Client

    response = Client().get("/")
    assert response.status_code == 200
    assert "不构成医学诊断" in response.content.decode("utf-8")


def test_upload_page_exposes_workbench_ui() -> None:
    from django.test import Client

    response = Client().get("/")
    body = response.content.decode("utf-8")
    assert response.status_code == 200
    assert 'class="app-shell"' in body
    assert 'class="dropzone"' in body
    assert "选择眼底图像" in body
    assert "模型已就绪" in body


def test_web_defaults_to_trained_idrid_artifact_when_present() -> None:
    from django.conf import settings

    assert settings.EYE_DEMO_ARTIFACT.name == "idrid_resnet_model.json"


def test_upload_page_rejects_non_image_without_crashing(tmp_path: Path, monkeypatch) -> None:
    from django.test import Client

    monkeypatch.setenv("EYE_DEMO_ARTIFACT_DIR", str(tmp_path))
    response = Client().post("/", {"image": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 200
    assert "JPG/PNG" in response.content.decode("utf-8") or "图像" in response.content.decode("utf-8")
