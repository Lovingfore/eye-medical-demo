"""预测应用 URL 路由。

# 【技术栈】Django URL dispatcher 将首页上传、结果页和历史页映射到 views.py；
# 模型本身不在路由层实现，便于后续替换推理后端。
"""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("result/", views.result, name="result"),
    path("history/", views.history, name="history"),
]
