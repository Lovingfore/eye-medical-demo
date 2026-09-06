"""Django 项目级路由入口，将根路径交给 predictor 应用处理。"""

from django.urls import include, path

urlpatterns = [path("", include("predictor.urls"))]
