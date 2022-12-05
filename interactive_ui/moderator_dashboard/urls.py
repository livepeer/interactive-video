from django.urls import include, path
from django.conf.urls import url
from . import views

app_name = "moderator_dashboard"

urlpatterns = [
    # ex: /moderator/
    path('', views.index, name='index'),
    # ex: /moderator/facelist
    path('facelist/', views.facelist, name="facelist"),    
    # ex: /moderator/facelist/10
    path('facelist/<int:face_id>/', views.facedetail, name='facedetail'),
    # ex: /moderator/facelist/10
    path('facelist/facedelete/<int:face_id>', views.facedelete, name='facedelete'),
    path('facelist/<int:face_id>/<str:sample_id>/<str:name>/<str:action>/<str:metadata>/', views.faceedit, name='faceedit'),
]