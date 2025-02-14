# blog/urls.py
from django.urls import path, include
from . import views

urlpatterns = [

    path('pool_settings/', views.pool_settings, name='pool_settings'),
    path('', include('allauth.urls')),
]
