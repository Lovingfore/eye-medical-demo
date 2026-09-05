from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []
    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("image_name", models.CharField(max_length=255)),
                ("class_name", models.CharField(max_length=32)),
                ("confidence", models.FloatField()),
                ("probabilities_json", models.JSONField(default=dict)),
                ("inference_ms", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
