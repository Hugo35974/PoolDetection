import json
import os
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .detector import SwimmingPoolDetector, get_detector, video_feed

# Chemin racine du projet
BASE_PATH = Path(__file__).parent.resolve()

@login_required
def video_stream(request,video_path=None):
    """
    Vue pour diffuser le flux vidéo.
    
    Args:
        request: Requête HTTP
        
    Returns:
        Réponse HTTP en streaming "rtsp://admin:123456@172.50.4.1:554/live/ch0"
    """
    
    if video_path is None:
        video_path = os.path.join(BASE_PATH, "data/video/eshan.mp4")
        
    return StreamingHttpResponse(
        video_feed(video_path), 
        content_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Accel-Buffering": "yes"}
    )


@login_required
def pool_settings(request):
    """
    Vue pour la page de paramètres.
    
    Args:
        request: Requête HTTP
        
    Returns:
        Rendu du template
    """
    detector = get_detector()
    context = {
        'expansion_factor': detector.expansion_factor,
        'size_threshold': detector.size_threshold,
        'calibration_active': detector.calibration,
    }
    return render(request, 'blog/pool_setting.html', context)


@csrf_exempt
def set_params(request):
    """
    Vue pour mettre à jour les paramètres.
    
    Args:
        request: Requête HTTP POST
        
    Returns:
        Réponse JSON
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            detector = get_detector()
            
            # Mise à jour des paramètres
            detector.expansion_factor = float(data.get("expansion_factor", 1.2))
            detector.size_threshold = int(data.get("size_threshold", 100))
            
            # Recalculer le masque si nécessaire
            if detector.pool_mask is not None:
                detector.calculate_mask()
                
            return JsonResponse({"status": "success", "message": "Paramètres mis à jour"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    
    return JsonResponse({"status": "error", "message": "Méthode non autorisée"}, status=405)


@csrf_exempt
def start_calibration(request):
    """
    Vue pour démarrer la calibration.
    
    Args:
        request: Requête HTTP POST
        
    Returns:
        Réponse JSON
    """
    if request.method == 'POST':
        try:
            detector = get_detector()
            detector.grid = False
            detector.draw = True
            detector.calibration = True
            detector.pool_detected = False
            
            return JsonResponse({"status": "success", "message": "Calibration démarrée"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    
    return JsonResponse({"status": "error", "message": "Méthode non autorisée"}, status=405)


@csrf_exempt
def stop_calibration(request):
    """
    Vue pour arrêter la calibration.
    
    Args:
        request: Requête HTTP POST
        
    Returns:
        Réponse JSON
    """
    if request.method == 'POST':
        try:
            detector = get_detector()
            detector.draw = False
            detector.calibration = False
            
            # Vérification de l'état de la calibration
            calibrated_points = sum(1 for factor in detector.scale_factors.values() if factor is not None)
            total_points = len(detector.grid_points)
            
            calibration_status = {
                "calibrated_points": calibrated_points,
                "total_points": total_points,
                "percentage": round(calibrated_points / total_points * 100) if total_points > 0 else 0
            }
            
            return JsonResponse({
                "status": "success", 
                "message": "Calibration arrêtée",
                "calibration": calibration_status
            })
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    
    return JsonResponse({"status": "error", "message": "Méthode non autorisée"}, status=405)