# blog/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import CustomUser, PoolSettings

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']

class PoolSettingsForm(forms.ModelForm):
    class Meta:
        model = PoolSettings
        fields = ['pool_length', 'pool_width', 'security_zone', 'adult_size', 'alert_email']
