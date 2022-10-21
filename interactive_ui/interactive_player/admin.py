from django.contrib import admin
from .models import UserFeedback

# Register your models here.
@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['user', 'feedback', 'time_window_id', 'created']