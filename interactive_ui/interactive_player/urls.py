from django.urls import include, path
from django.conf.urls import url
from . import views
from django.contrib.auth import views as auth_views

app_name = "interactive_player"

urlpatterns = [
    path('', views.index, name="index"),
    url(r'^feedback', views.UserFeedbackView, name='feedback'),
    url(r'^downcount', views.downcount, name='downcount'),

]