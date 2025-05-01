from django.db import models

class VideoDetection(models.Model):
    timestamp   = models.DateTimeField(auto_now_add=True)
    input_file  = models.CharField(max_length=255)
    output_file = models.CharField(max_length=255)
    confidence  = models.FloatField()