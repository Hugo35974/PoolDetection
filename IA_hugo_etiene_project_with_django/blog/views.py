import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

Main_path = Path(__file__).parents[0]

class SwimmingPoolDetector:
    def __init__(self, model_path: str, person_model_path: str,
                 expansion_factor: float = 1.1, size_threshold: int = 100,
                 device="cpu", known_height=1.70):
        self.pool_model = YOLO(model_path)
        self.person_model = YOLO(person_model_path)

        self.pool_detected = False
        self.pool_mask = None
        self.dilated_mask = None
        self.expansion_factor = expansion_factor
        self.size_threshold = size_threshold
        self.known_height = known_height
        self.grid_points = []
        self.scale_factors = {}
        self.max_depth_points = 100
        self.grid = False
        

    def create_grid(self, frame, grid_size=(10, 10), expansion_factor=3):
        """Crée une grille de points autour du contour de la piscine avec expansion."""
        if self.pool_mask is None:
            print("Piscine non détectée. Impossible de créer la grille.")
            return

        # Dilater le masque de la piscine pour créer une zone autour
        h, w = self.pool_mask.shape
        kernel_size = max(5, int(min(w, h) * (expansion_factor - 1) * 0.1))  # Adapté à la taille
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(self.pool_mask, kernel, iterations=1)

        # Trouver les contours du masque dilaté
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Aucun contour détecté pour la zone dilatée.")
            return

        # Récupérer le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Récupérer la bounding box élargie
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Définir les intervalles pour la grille
        x_interval = w // grid_size[1]
        y_interval = h // grid_size[0]

        # Générer les points de la grille autour du masque
        self.grid_points = []
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                x_point = x + col * x_interval + x_interval // 2
                y_point = y + row * y_interval + y_interval // 2

                # Vérifier que le point est DANS la zone dilatée mais PAS dans la piscine
                if dilated_mask[y_point, x_point] > 0 and self.pool_mask[y_point, x_point] == 0:
                    self.grid_points.append((x_point, y_point))
                    self.scale_factors[(x_point, y_point)] = None

        self.grid = True

    def draw_calibration_points(self, frame):
        """Dessine les points de calibration sur l'image."""
        for point in self.grid_points:
            color = (0, 0, 255) if self.scale_factors[point] is None else (0, 255, 0)
            cv2.circle(frame, point, 5, color, -1)

    def update_scale_factor(self, frame, detected_feet_box):
        """Met à jour le facteur d'échelle si les pieds sont sur un point de calibration."""
        x1, y1, x2, y2 = detected_feet_box
        feet_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.scale_factors_dict = {}
        for point in self.grid_points:
            if self.is_near_point(feet_center, point) and self.scale_factors[point] is None:
                detected_height = y2 - y1
                self.scale_factors[point] = self.known_height / detected_height
                print(f"Point {point} validé avec un facteur d'échelle de {self.scale_factors[point]}")

    #     self.scale_factors_dict = {str(key): value for key, value in self.scale_factors.items()}

    #     # Sauvegarde dans un fichier JSON
    #     with open("scale_factors_grid.json", "w") as f:
    #         json.dump(self.scale_factors_dict, f, indent=4)

    #     print("Grille et scale factors sauvegardés dans scale_factors_grid.json")

    def is_near_point(self, feet_center, point, threshold=10):
        """Vérifie si les pieds sont proches d'un point de calibration."""
        distance = np.linalg.norm(np.array(feet_center) - np.array(point))
        return distance < threshold

    def is_calibration_complete(self):
        """Vérifie si tous les points ont été validés."""
        return all(factor is not None for factor in self.scale_factors.values())

    def get_calibration_coefficient(self, y_position: int, image_height: int) -> float:
        """
        Retourne un coefficient de calibration linéaire basé sur la position verticale de l'objet
        dans l'image. Plus l'objet est bas (près de la caméra), plus le coefficient sera élevé.
        """
        relative_position = y_position / image_height
        calibration_factor = 1 - relative_position  # Inversement proportionnel à la hauteur (bas -> haut)
        return calibration_factor

    def adjust_size_based_on_perspective(self, x1, y1, x2, y2, image_height: int) -> float:
        """Ajuste la taille d'une personne en fonction de la perspective."""
        initial_size = y2 - y1
        perspective_factor = self.get_calibration_coefficient(y1, image_height)
        adjusted_size = initial_size * perspective_factor

        return adjusted_size

    def classify_person(self, adjusted_size: float) -> str:
        """Classe la personne comme 'enfant' ou 'adulte' selon sa taille ajustée."""
        if adjusted_size < self.size_threshold:
            return "Enfant"
        else:
            return "Adulte"

    def detect_person_and_classify(self, frame: np.ndarray) -> np.ndarray:
        result = self.person_model.predict(frame, verbose=False)[0]
        annotated_image = frame.copy()

        image_height, image_width, _ = frame.shape

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == 0 and conf > 0.5:  # Vérifie si c'est une personne avec une confiance élevée
                    x1, y1, x2, y2 = map(int, box)
                    height = y2 - y1
                    width = x2 - x1

                    # Ajustement de la taille
                    adjusted_size = self.adjust_size_based_on_perspective(x1, y1, x2, y2, image_height)

                    # Classification
                    person_class = self.classify_person(adjusted_size)

                    # Choisir la couleur du cadre en fonction de la classification
                    if person_class == "Enfant":
                        rectangle_color = (0, 0, 255)  # Rouge pour enfant
                    else:
                        rectangle_color = (0, 255, 0)  # Vert pour adulte

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), rectangle_color, 2)
                    cv2.putText(annotated_image, f"Taille : {adjusted_size:.2f} m",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f"{person_class}",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                    # Mettre à jour le facteur d'échelle si les pieds sont détectés
                    self.update_scale_factor(annotated_image, (x1, y1, x2, y2))

        return annotated_image

    def detect_pool_and_danger_zone(self, frame: np.ndarray) -> np.ndarray:
        if self.pool_detected:
            return frame

        result = self.pool_model.predict(frame)[0]
        annotated_image = frame.copy()

        for box, mask, conf, cls in zip(result.boxes.xyxy, result.masks.data, result.boxes.conf, result.boxes.cls):
            if int(cls) == 0 and conf > 0.3:
                x1, y1, x2, y2 = map(int, box)
                mask = mask.cpu().numpy().squeeze()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                mask_colored = np.zeros_like(frame, dtype=np.uint8)
                mask_colored[:, :, 1] = mask_binary

                annotated_image = cv2.addWeighted(annotated_image, 1, mask_colored, 1, 0)
                # cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                self.pool_detected = True
                self.pool_mask = mask_binary
                self.calculate_mask()

        return annotated_image

    def calculate_mask(self):
        kernel_size = int(self.expansion_factor * 10)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.dilated_mask = cv2.dilate(self.pool_mask, kernel, iterations=1)

    def draw_contours(self, frame: np.ndarray) -> np.ndarray:
        if self.pool_detected and self.dilated_mask is not None:
            border_image = np.zeros_like(frame, dtype=np.uint8)
            border_image[:, :, 2] = self.dilated_mask
            frame = cv2.addWeighted(frame, 1, border_image, 1, 0)
        return frame

# Code principal pour l'exécution de la vidéo
video_path = os.path.join(Main_path, "data/video/TombeDedans.mp4")
detector = SwimmingPoolDetector(
    model_path=os.path.join(Main_path, 'data/model/best.pt'),
    person_model_path=os.path.join(Main_path, 'data/model/yolo11n.pt'),
    expansion_factor=1.2, device="mps"
)

cap = cv2.VideoCapture(video_path)

def video_feed():
    frame_count = 0
    modulo = 2
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if frame_count % modulo == 0:
            if not detector.grid :
                detector.create_grid(frame)
                print("Création de grille")
            detector.draw_calibration_points(frame)  # Dessine les points de calibration
            frame = detector.detect_pool_and_danger_zone(frame)
            frame = detector.detect_person_and_classify(frame)

        frame = detector.draw_contours(frame)
        frame = cv2.resize(frame, (1280, 720))
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1

@login_required
def video_stream(request):
    return StreamingHttpResponse(video_feed(), content_type="multipart/x-mixed-replace; boundary=frame", headers={"X-Accel-Buffering": "yes"})

@login_required
def pool_settings(request):
    return render(request, 'blog/pool_setting.html')

@csrf_exempt
def set_params(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        detector.expansion_factor = float(data.get("expansion_factor", 1.2))
        detector.size_threshold = int(data.get("size_threshold", 100))

        if detector.pool_mask is not None:
            detector.calculate_mask()

        return JsonResponse({"message": "Paramètres mis à jour"})
