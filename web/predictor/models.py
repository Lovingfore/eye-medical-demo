from django.db import models


class Prediction(models.Model):
    image_name = models.CharField(max_length=255)
    class_name = models.CharField(max_length=32)
    confidence = models.FloatField()
    probabilities_json = models.JSONField(default=dict)
    inference_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
