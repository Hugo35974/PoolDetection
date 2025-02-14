# blog/urls.py
from django.urls import include, path

from . import views

urlpatterns = [

    path('pool_settings/', views.pool_settings, name='pool_settings'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('set_params/', views.set_params, name='set_params'),
    path('', include('allauth.urls')),
]
