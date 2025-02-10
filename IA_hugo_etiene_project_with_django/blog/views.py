# blog/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, PoolSettingsForm
from .models import PoolSettings

# Vue pour les paramètres de la piscine
@login_required
def pool_settings(request):
    settings = PoolSettings.objects.get(user=request.user)

    if request.method == 'POST':
        form = PoolSettingsForm(request.POST, instance=settings)
        if form.is_valid():
            form.save()
            return render(request, 'pool_settings.html', {'form': form, 'success': 'Paramètres sauvegardés avec succès.'})
    else:
        form = PoolSettingsForm(instance=settings)

    return render(request, 'allauth/base.html', {'form': form})