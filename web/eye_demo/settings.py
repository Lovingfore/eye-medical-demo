from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
# 生产环境必须通过环境变量覆盖默认密钥；默认值只方便本地开发。
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "eye-demo-local-only")
DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver"
    ).split(",")
    if host.strip()
]
render_hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if render_hostname and render_hostname not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(render_hostname)
ROOT_URLCONF = "eye_demo.urls"
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "predictor.apps.PredictorConfig",
]
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]
WSGI_APPLICATION = "eye_demo.wsgi.application"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": Path(os.environ.get("EYE_DEMO_DB", str(BASE_DIR / "demo.sqlite3"))),
    }
}
LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
_trained_artifact = BASE_DIR.parent / "artifacts" / "idrid_resnet_model.json"
_demo_artifact = BASE_DIR.parent / "artifacts" / "model.json"
EYE_DEMO_ARTIFACT = Path(os.environ.get("EYE_DEMO_ARTIFACT", str(_trained_artifact if _trained_artifact.exists() else _demo_artifact)))
EYE_DEMO_UPLOAD_DIR = Path(os.environ.get("EYE_DEMO_UPLOAD_DIR", str(BASE_DIR.parent / "artifacts" / "uploads")))
# Django 的内存上传阈值与视图中的 8 MB 业务限制保持一致。
FILE_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
