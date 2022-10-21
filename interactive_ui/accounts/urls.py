from django.urls import include, path
from django.conf.urls import url
from . import views
from django.contrib.auth import views as auth_views

app_name = "accounts"

urlpatterns = [
    path('login', auth_views.LoginView.as_view(), name='login'),
]




