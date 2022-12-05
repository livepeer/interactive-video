from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from interactive_player.models import UserFeedback
from django.db.models import Count
import time
from base64 import b64decode
from . import dbconnect
from django.http import HttpResponseRedirect
from django.urls import path, reverse

# Create your views here.
def check_admin(user):
    return user.is_superuser

@user_passes_test(check_admin)
def index(request):
    window_id = int(time.time() / 30)
    feedbacks = UserFeedback.objects.filter(time_window_id=window_id).values('feedback').annotate(fcount=Count('feedback')).order_by('fcount')[:10]
    return render(request, 'moderator_dashboard/dashboard.html',
                  {
                     'feedbacks': feedbacks
                  })
   
@user_passes_test(check_admin)
def feedback_sort(request):
    window_id = int(time.time() / 30)
    return render(request, 'moderator_dashboard/dashboard.html',
                  {
                  })
@user_passes_test(check_admin)
def facelist(request):
    window_id = int(time.time() / 30)
    facelist = dbconnect.get_facelist()
    return render(request, 'moderator_dashboard/facelist.html',
                  {
                     'facelist': facelist
                  })

@user_passes_test(check_admin)
def facedetail(request, face_id):
    window_id = int(time.time() / 30)
    face = dbconnect.get_facedetail(face_id)
    return render(request, 'moderator_dashboard/facedetail.html',
                  {
                     'face': face
                  })
@user_passes_test(check_admin)
def facedelete(request, face_id):
    window_id = int(time.time() / 30)
    dbconnect.delete_facedetail(face_id)
    return redirect(reverse('moderator_dashboard:facelist'))
    
@user_passes_test(check_admin)
def faceedit(request, face_id, sample_id, name, action, metadata):
    window_id = int(time.time() / 30)
    metadatadec = b64decode(metadata).decode('utf-8')
    dbconnect.update_facedetail(face_id, sample_id, name, action, metadatadec)
    return redirect(reverse('moderator_dashboard:facelist'))
    