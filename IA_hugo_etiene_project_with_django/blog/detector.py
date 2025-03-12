import logging
import os
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from django.core.mail import EmailMessage
from django.core.mail import send_mail
from django.conf import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pool_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Chargement des modèles
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics n'est pas installé. Exécutez 'pip install ultralytics'.")
    raise

# Chemin racine
BASE_PATH = Path(__file__).parent.resolve()

# Configuration constantes
DEFAULT_EXPANSION_FACTOR = 2.5
DEFAULT_SIZE_THRESHOLD = 130
DEFAULT_GRID_SIZE = (40, 40)
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
POOL_CLASS_ID = 0
EMAIL_THRESHOLD = 40  # Nombre de frames consécutives avec enfant pour déclencher l'alerte
RECIPIENT_LIST = ["@isen-ouest.yncrea.fr"]  # À configurer dans settings.py


class SwimmingPoolDetector:
    """Détecteur de piscine et classificateur de personnes avec mesures de sécurité."""
    
    def __init__(
        self, 
        model_path: str, 
        person_model_path: str,
        expansion_factor: float = DEFAULT_EXPANSION_FACTOR, 
        size_threshold: int = DEFAULT_SIZE_THRESHOLD,
        device: str = "cpu", 
        known_height: int = 170
    ):
        """
        Initialise le détecteur de piscine avec les modèles YOLO.
        
        Args:
            model_path: Chemin vers le modèle YOLO de détection de piscine
            person_model_path: Chemin vers le modèle YOLO de détection de personnes
            expansion_factor: Facteur d'expansion pour la zone de danger autour de la piscine
            size_threshold: Seuil de taille pour distinguer un enfant d'un adulte
            device: Appareil pour l'inférence ("cpu", "cuda", "mps")
            known_height: Taille connue de référence en cm
        """
        try:
            self.pool_model = YOLO(model_path)
            self.person_model = YOLO(person_model_path)
            logger.info(f"Modèles chargés avec succès: {model_path} et {person_model_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            raise

        # États et configurations
        self.pool_detected = False
        self.pool_mask = None
        self.dilated_mask = None
        self.expansion_factor = expansion_factor
        self.size_threshold = size_threshold
        self.known_height = known_height
        self.device = device
        
        # Calibration et grille
        self.grid_points = []
        self.scale_factors = {}
        self.max_depth_points = 100
        self.grid = False
        self.draw = True
        self.calibration = True
        
        # Détection et alertes
        self.child_detected_frames = 0
        self.email_sent = False
        
        logger.info("SwimmingPoolDetector initialisé avec succès")

    def send_alert_email(self, frame) -> None:
        """Envoie un email d'alerte avec la frame en pièce jointe."""
        subject = "ALERTE: Enfant détecté près de la piscine"
        message = "Un enfant a été détecté près de la piscine. Veuillez vérifier la pièce jointe."
        
        try:
            # Convertir la frame (image OpenCV) en fichier binaire
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            # Créer l'email
            email = EmailMessage(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                RECIPIENT_LIST,
            )
            
            # Ajouter l'image en pièce jointe
            email.attach("alerte_piscine.jpg", image_bytes, "image/jpeg")
            
            # Envoyer l'email
            email.send(fail_silently=False)
            
            logger.info("Alerte envoyée par email avec succès avec image jointe")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email d'alerte: {e}")

    def create_grid(self, frame: np.ndarray, grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE) -> None:
        """
        Crée une grille de points autour du contour de la piscine pour la calibration.
        
        Args:
            frame: Image d'entrée
            grid_size: Taille de la grille (lignes, colonnes)
        """
        if self.pool_mask is None:
            logger.warning("Impossible de créer la grille: piscine non détectée")
            return

        contours, _ = cv2.findContours(self.dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("Aucun contour détecté pour la zone dilatée")
            return

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        x_interval = w // grid_size[1]
        y_interval = h // grid_size[0]

        # Réinitialiser les points de grille
        self.grid_points = []
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                x_point = x + col * x_interval + x_interval // 2
                y_point = y + row * y_interval + y_interval // 2
                
                # Ne conserver que les points dans la zone de danger mais pas dans la piscine
                if self.dilated_mask[y_point, x_point] > 0 and self.pool_mask[y_point, x_point] == 0:
                    self.grid_points.append((x_point, y_point))
                    self.scale_factors[(x_point, y_point)] = None

        self.grid = True
        logger.info(f"Grille créée avec {len(self.grid_points)} points de calibration")

    def draw_calibration_points(self, frame: np.ndarray) -> None:
        """
        Dessine les points de calibration sur l'image.
        
        Args:
            frame: Image sur laquelle dessiner les points
        """
        for point in self.grid_points:
            color = (0, 0, 255) if self.scale_factors[point] is None else (0, 255, 0)
            cv2.circle(frame, point, 5, color, -1)

    def update_scale_factor(self, detected_feet_box: Tuple[int, int, int, int]) -> None:
        """
        Met à jour le facteur d'échelle si les pieds sont sur un point de calibration.
        
        Args:
            detected_feet_box: Boîte englobante (x1, y1, x2, y2) de la personne détectée
        """
        x1, y1, x2, y2 = detected_feet_box
        feet_center = ((x1 + x2) // 2, y2)
        
        for point in self.grid_points:
            if self.is_near_point(feet_center, point) and self.scale_factors[point] is None:
                detected_height = y2 - y1
                self.scale_factors[point] = self.known_height / detected_height
                logger.debug(f"Point {point} validé avec facteur d'échelle: {self.scale_factors[point]:.2f}")

    def is_near_point(self, feet_center: Tuple[int, int], point: Tuple[int, int], threshold: int = 100) -> bool:
        """
        Vérifie si les pieds sont proches d'un point de calibration.
        
        Args:
            feet_center: Centre des pieds (x, y)
            point: Point de calibration (x, y)
            threshold: Distance maximale en pixels
            
        Returns:
            True si la distance est inférieure au seuil
        """
        distance = np.linalg.norm(np.array(feet_center) - np.array(point))
        return distance < threshold

    def adjust_size_based_on_perspective(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Ajuste la taille d'une personne en fonction du point de calibration le plus proche.
        
        Args:
            x1, y1, x2, y2: Coordonnées de la boîte englobante
            
        Returns:
            Taille ajustée en fonction de la perspective
        """
        height = y2 - y1
        feet_center = ((x1 + x2) // 2, y2)

        # Pas de points de calibration
        if not self.grid_points:
            return height
        
        # Trouver le point de calibration le plus proche
        nearest_point = min(self.grid_points, key=lambda p: np.linalg.norm(np.array(feet_center) - np.array(p)))
        
        # Récupérer le facteur d'échelle
        scale_factor = self.scale_factors.get(nearest_point)
        if scale_factor is None:
            # Si le point n'est pas calibré, utiliser la moyenne des facteurs existants ou 1.0
            calibrated_factors = [f for f in self.scale_factors.values() if f is not None]
            scale_factor = sum(calibrated_factors) / len(calibrated_factors) if calibrated_factors else 1.0
            
        return scale_factor * height

    def classify_person(self, adjusted_size: float) -> str:
        """
        Classe la personne comme 'enfant' ou 'adulte' selon sa taille ajustée.
        
        Args:
            adjusted_size: Taille ajustée de la personne
            
        Returns:
            "Enfant" ou "Adulte"
        """
        return "Enfant" if adjusted_size < self.size_threshold else "Adulte"

    def detect_person_and_classify(self, frame: np.ndarray) -> np.ndarray:
        """
        Détecte les personnes et les classifie comme enfants ou adultes.
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Image annotée avec les détections
        """
        result = self.person_model.predict(frame, verbose=False)[0]
        annotated_image = frame.copy()
        enfant_detecte = False
        
        if result.boxes and len(result.boxes) > 0:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calcul de la taille ajustée
                    adjusted_size = self.adjust_size_based_on_perspective(x1, y1, x2, y2)
                    person_class = self.classify_person(adjusted_size)
                    
                    # Couleur du rectangle selon la classification
                    rectangle_color = (0, 0, 255) if person_class == "Enfant" else (0, 255, 0)
                    
                    if person_class == "Enfant":
                        enfant_detecte = True
                    
                    # Annotation de l'image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), rectangle_color, 2)
                    # cv2.putText(annotated_image, f"Taille: {adjusted_size:.0f} cm",
                    #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f"{person_class}",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Mise à jour de la calibration si active
                    if self.calibration:
                        # self.send_alert_email()
                        self.update_scale_factor((x1, y1, x2, y2))

        # Gestion des alertes
        if enfant_detecte:
            self.child_detected_frames += 1
            if self.child_detected_frames > EMAIL_THRESHOLD and not self.email_sent:
                self.send_alert_email(frame)
                self.email_sent = True
                logger.warning("Alerte: enfant détecté pendant plus de 3 frames consécutives")
        else:
            self.child_detected_frames = 0
            # self.email_sent = False

        return annotated_image

    def detect_pool_and_danger_zone(self, frame: np.ndarray) -> np.ndarray:
        """
        Détecte la piscine et crée une zone de danger autour.
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Image annotée avec la piscine détectée
        """
        # Si la piscine est déjà détectée, retourner simplement l'image
        if self.pool_detected:
            return frame

        result = self.pool_model.predict(frame, verbose=False)[0]
        if result.masks is None:
            return frame

        annotated_image = frame.copy()
        if result:
            # Traiter les résultats de détection
            for box, mask, conf, cls in zip(result.boxes.xyxy, result.masks.data, result.boxes.conf, result.boxes.cls):
                if int(cls) == POOL_CLASS_ID and conf > 0.5:  # Classe 0 = piscine
                    
                    # Traitement du masque
                    mask = mask.cpu().numpy().squeeze()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Coloration du masque
                    mask_colored = np.zeros_like(frame, dtype=np.uint8)
                    mask_colored[:, :, 1] = mask_binary  # Canal vert

                    # Fusion avec l'image originale
                    annotated_image = cv2.addWeighted(annotated_image, 1, mask_colored, 0.5, 0)
                    
                    # Mise à jour de l'état
                    self.pool_detected = True
                    self.pool_mask = mask_binary
                    self.calculate_mask()
                    logger.info("Piscine détectée avec succès")

            return annotated_image

    def calculate_mask(self) -> None:
        """Calcule le masque dilaté pour la zone de danger."""
        h, w = self.pool_mask.shape
        kernel_size = max(5, int(min(w, h) * (self.expansion_factor - 1) * 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.dilated_mask = cv2.dilate(self.pool_mask, kernel, iterations=1)

    def draw_contours(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les contours de la zone de danger autour de la piscine.
        
        Args:
            frame: Image d'entrée
            
        Returns:
            Image avec contours
        """
        if self.pool_detected and self.dilated_mask is not None:
            # Créer une image pour le contour
            border_image = np.zeros_like(frame, dtype=np.uint8)
            border_image[:, :, 2] = self.dilated_mask  # Canal rouge
            
            # Trouver les contours de la zone de danger
            contours, _ = cv2.findContours(self.dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Dessiner uniquement le contour extérieur
            if contours:
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                
                # Ajouter un fond semi-transparent pour la zone de danger
                alpha = 0.3
                frame = cv2.addWeighted(frame, 1, border_image, alpha, 0)
                
        return frame
    
    def process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """
        Traite une frame vidéo en appliquant les détections et visualisations.
        
        Args:
            frame: Image d'entrée
            frame_count: Compteur de frames
            
        Returns:
            Image traitée
        """
        self.frame = frame
        # Optimisation: ne pas traiter toutes les frames
        modulo = 2
        if frame_count % modulo == 0:
            # Créer la grille si nécessaire
            if not self.grid and self.pool_detected:
                self.create_grid(frame)
                logger.debug("Grille de calibration créée")

            # Détection de piscine
            frame = self.detect_pool_and_danger_zone(frame)
    
            frame = self.detect_person_and_classify(frame)

        # Affichage des éléments visuels
        if self.draw:
            self.draw_calibration_points(frame)
            frame = self.draw_contours(frame)
        
        return frame


# Singleton pattern pour le détecteur
_detector_instance = None

def get_detector():
    """
    Fonction singleton pour récupérer l'instance unique du détecteur.
    
    Returns:
        Instance de SwimmingPoolDetector
    """
    global _detector_instance
    if _detector_instance is None:
            # Chemins relatifs depuis le fichier actuel
        _detector_instance = SwimmingPoolDetector(
            model_path=os.path.join(BASE_PATH, 'data/model/best.pt'),
            person_model_path=os.path.join(BASE_PATH, 'data/model/yolo11n.pt'),
            expansion_factor=DEFAULT_EXPANSION_FACTOR,
            device="mps"  # Utiliser "cpu" par défaut pour la compatibilité
        )
    return _detector_instance


def video_feed(video_path):
    """
    Générateur pour le flux vidéo avec détection.
    
    Args:
        video_path: Chemin de la vidéo (optionnel)
        
    Yields:
        Frames JPEG encodées pour le streaming HTTP
    """

    cap = cv2.VideoCapture(video_path)
    detector = get_detector()
    frame_count = 0
    modulo = 1

    logger.info("Connexion établie avec la caméra")
    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                logger.warning("Frame non lue ou vide, tentative de redémarrage...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.waitKey(1)
            time.sleep(1/30) 
            frame = cv2.resize(frame, (1280, 720))
            if frame_count % modulo == 0:
                frame = detector.process_frame(frame, frame_count)
            
            # Redimensionner pour l'affichage web
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Envoyer la frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
    except Exception as e:
        logger.error(f"Erreur dans le flux vidéo: {e}")
    
    finally:
        cap.release()
        logger.info("Fin du flux vidéo")