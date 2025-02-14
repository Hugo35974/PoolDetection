# blog/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    """
    Modèle utilisateur personnalisé pour stocker les informations utilisateur.
    """
    pass

class PoolSettings(models.Model):

    #user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name="pool_settings", null=True, blank=True)
    pool_length = models.FloatField(default=8.0)
    pool_width = models.FloatField(default=4.0)
    security_zone = models.FloatField(default=1.0)
    adult_size = models.FloatField(default=1.80)
    alert_email = models.EmailField(blank=True, null=True)

    def __str__(self):
        return f"Paramètres de {self.user.username}"