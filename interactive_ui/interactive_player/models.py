from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UserFeedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feedback = models.CharField(max_length=1024, default='', blank=True, null=True)
    time_window_id = models.IntegerField()
    created = models.DateTimeField(auto_now_add=True)   
