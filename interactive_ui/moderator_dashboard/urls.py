from django.urls import include, path
from django.conf.urls import url
from . import views

app_name = "moderator_dashboard"

urlpatterns = [
    path('', views.index, name="index"),
]