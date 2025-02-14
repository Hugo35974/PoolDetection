import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

Main_path = Path(__file__).parents[0]
print(Main_path)
class SwimmingPoolDetector:
    def __init__(self, model_path: str, person_model_path: str, expansion_factor: float = 1.1, size_threshold: int = 100, device="cpu"):
        self.pool_model = YOLO(model_path)
        self.person_model = YOLO(person_model_path)
        self.pool_detected = False
        self.pool_box = None
        self.pool_mask = None
        self.dilated_mask = None 
        self.expansion_factor = expansion_factor
        self.size_threshold = size_threshold
        self.device = device

    def detect_pool_and_danger_zone(self, frame: np.ndarray) -> np.ndarray:
        if self.pool_detected:
            return frame

        result = self.pool_model.predict(frame)[0]
        annotated_image = frame.copy()

        for box, mask, conf, cls in zip(result.boxes.xyxy, result.masks.data, result.boxes.conf, result.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                mask = mask.cpu().numpy().squeeze()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                mask_colored = np.zeros_like(frame, dtype=np.uint8)
                mask_colored[:, :, 1] = mask_binary

                annotated_image = cv2.addWeighted(annotated_image, 1, mask_colored, 1, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                self.pool_detected = True
                self.pool_box = (x1, y1, x2, y2)
                self.pool_mask = mask_binary

                # Calculer et enregistrer le masque dilaté
                kernel_size = int(self.expansion_factor * 10)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.dilated_mask = cv2.dilate(self.pool_mask, kernel, iterations=1)

        return annotated_image

    def detect_person_and_classify(self, frame: np.ndarray) -> np.ndarray:
        result = self.person_model.predict(frame, verbose=False)[0]
        annotated_image = frame.copy()

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    height = y2 - y1
                    label = "Adulte" if height > self.size_threshold else "Enfant"
                    color = (0, 0, 255)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_image, f"{label} {conf:.2f}",
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 2)

        return annotated_image

    def draw_contours(self, frame: np.ndarray) -> np.ndarray:
        if self.pool_detected and self.dilated_mask is not None:
            border_image = np.zeros_like(frame, dtype=np.uint8)
            border_image[:, :, 2] = self.dilated_mask
            frame = cv2.addWeighted(frame, 1, border_image, 1, 0)
        return frame

video_path = os.path.join(Main_path,"data/video/93705-642181946.mp4")
detector = SwimmingPoolDetector(
    model_path=os.path.join(Main_path,'data/model/best.pt'),
    person_model_path=os.path.join(Main_path,'data/model/yolo11n.pt'),
    expansion_factor=1.2,
    size_threshold=100, device="mps"
)
cap = cv2.VideoCapture(video_path)

def video_feed():
    frame_count = 0
    modulo = 5
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if frame_count % modulo == 0:
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

        # Recalculer le masque dilaté si le facteur d'expansion change
        if detector.pool_mask is not None:
            kernel_size = int(detector.expansion_factor * 10)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            detector.dilated_mask = cv2.dilate(detector.pool_mask, kernel, iterations=1)

        return JsonResponse({"message": "Paramètres mis à jour"})
