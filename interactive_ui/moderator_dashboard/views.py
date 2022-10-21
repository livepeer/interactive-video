from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from interactive_player.models import UserFeedback
from django.db.models import Count
import time

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
