from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from .models import UserFeedback
import time

# Create your views here.
def index(request):
    print(round(time.time()))
    return render(request, 'interactive_player/index.html',
                  {
                  })


def downcount(request):
    remaining_time = 30 - int(time.time() % 30)
    window_id = int(time.time() / 30)
    return JsonResponse({'remaining':remaining_time, 'window_id': window_id}, status=200)


@login_required
def UserFeedbackView(request):
    feedback = request.POST['feedback']
    window_id = int(request.POST['window_id'])
    UserFeedback.objects.create(user=request.user, feedback=feedback, time_window_id=window_id)

    return HttpResponse('success', status=200)


