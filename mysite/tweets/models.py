from django.db import models

# Create your models here.

class Tweet(models.Model):

    text = models.TextField()
    timestamp = models.DateTimeField()

