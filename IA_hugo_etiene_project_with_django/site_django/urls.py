# site_django/urls.py
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('site/', include('blog.urls')),
    path('', lambda request: redirect('site/login/'))
]