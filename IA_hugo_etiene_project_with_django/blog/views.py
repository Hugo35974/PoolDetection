# blog/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, PoolSettingsForm
from .models import PoolSettings

# Vue pour les paramètres de la piscine
@login_required
def pool_settings(request):
    return render(request, 'blog/pool_setting.html')