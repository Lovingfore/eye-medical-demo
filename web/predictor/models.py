"""Django ORM 数据模型。

# 【技术栈】Django ORM + SQLite；Prediction 只保存上传文件名、类别概率、置信度、
# 推理耗时和时间，不保存训练标签，也不会把 Web 上传图片自动加入训练集。
"""

from django.db import models


class Prediction(models.Model):
    # 【预测历史】每次上传推理成功后写入一条记录，供 /history/ 页面展示。
    image_name = models.CharField(max_length=255)
    class_name = models.CharField(max_length=32)
    confidence = models.FloatField()
    probabilities_json = models.JSONField(default=dict)
    inference_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
