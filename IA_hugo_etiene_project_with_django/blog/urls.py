# blog/urls.py
from django.urls import path, include
from . import views

urlpatterns = [

    path('pool-settings/', views.pool_settings, name='pool-settings'),
    path('', include('allauth.urls')),
]