import os
import shutil
import math
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FallDetector:
    def __init__(self, model_path="pose_landmarker_heavy.task"):
        self._init_error = None
        try:
            self._init_detector(model_path)
        except Exception as e:
            self._init_error = str(e)
            print(f"[FallDetector] Не удалось инициализировать детектор: {e}")

    def _init_detector(self, model_path="pose_landmarker_heavy.task"):
        """
        Инициализация детектора падений на базе MediaPipe PoseLandmarker (Tasks API).
        С автозагрузкой файла модели, если он отсутствует.
        """
        # Автоматическая загрузка модели, если её нет в директории
        if not os.path.exists(model_path):
            print(f"[MediaPipe] Файл {model_path} не найден. Скачивание модели...")
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("[MediaPipe] Файл модели успешно загружен!")
            except Exception as e:
                raise FileNotFoundError(
                    f"Не удалось автоматически загрузить модель. "
                    f"Скачайте её вручную по ссылке:\n{model_url}\nОшибка: {e}"
                )

        model_path = os.path.abspath(model_path)
        if not model_path.isascii():
            ascii_dir = r"C:\MediaPipeModels"
            ascii_path = os.path.join(ascii_dir, os.path.basename(model_path))
            os.makedirs(ascii_dir, exist_ok=True)
            if not os.path.exists(ascii_path):
                try:
                    shutil.copy2(model_path, ascii_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Не удалось подготовить модель для загрузчика MediaPipe: {e}"
                    )
            if os.path.exists(ascii_path):
                model_path = ascii_path

        # Настройка параметров MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Константы для отрисовки скелета (связи между ключевыми точками)
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
            (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]

    def _calculate_angle(self, p1, p2):
        """
        Вычисление угла наклона линии между двумя точками относительно горизонтали.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        angle_rad = math.atan2(abs(dy), abs(dx))
        return math.degrees(angle_rad)

    def process_frame(self, frame_bgr, timestamp_ms=None):
        try:
            return self._process_frame_impl(frame_bgr, timestamp_ms)
        except Exception as e:
            print(f"[FallDetector] Необработанная ошибка кадра: {e}")
            try:
                debug_path = os.path.join(os.path.dirname(__file__), "bad_frame_debug.jpg")
                cv2.imwrite(debug_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print(f"[FallDetector] Сохранил проблемный кадр в: {debug_path}")
            except Exception:
                pass
            return frame_bgr.copy(), False, 0.0, 0.0

    def _process_frame_impl(self, frame_bgr, timestamp_ms=None):
        """
        Обработка кадра BGR: поиск скелета, определение падения и отрисовка.
        
        Возвращает:
            processed_frame (np.ndarray): Кадр с отрисованным скелетом и статусом.
            is_fall (bool): Флаг обнаружения падения.
            angle (float): Угол наклона туловища.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Конвертация BGR в RGB и подготовка MediaPipe Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        is_fall = False
        body_angle = 0.0
        fall_confidence = 0.0
        output_frame = frame_bgr.copy()

        # Детекция ключевых точек
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"[MediaPipe] Ошибка детекции: {e}")
            return output_frame, is_fall, body_angle, fall_confidence

        if detection_result.pose_landmarks:
            # Берем первого найденного человека
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame_bgr.shape

            # Точки плеч (11 - левое, 12 - правое) и бедер (23 - левое, 24 - правое)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Вычисляем центры плечевого пояса и таза
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2

            class Point:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            shoulder_center = Point(shoulder_center_x, shoulder_center_y)
            hip_center = Point(hip_center_x, hip_center_y)

            # Расчет угла наклона корпуса относительно горизонтали
            body_angle = self._calculate_angle(shoulder_center, hip_center)

            # Критерий падения: угол наклона корпуса к горизонтали меньше 45 градусов
            # (в нормальном положении стоя/сидя угол ближе к 90 градусам)
            if body_angle < 45.0:
                is_fall = True

            # Отрисовка связей (костей)
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    pt1 = landmarks[start_idx]
                    pt2 = landmarks[end_idx]

                    # Пропускаем отрисовку, если видимость точек слишком низкая
                    if getattr(pt1, 'presence', 1.0) < 0.3 or getattr(pt2, 'presence', 1.0) < 0.3:
                        continue

                    px1, py1 = int(pt1.x * w), int(pt1.y * h)
                    px2, py2 = int(pt2.x * w), int(pt2.y * h)

                    line_color = (0, 0, 255) if is_fall else (0, 255, 0)
                    cv2.line(output_frame, (px1, py1), (px2, py2), line_color, 2)

            # Отрисовка ключевых точек (суставов)
            for lm in landmarks:
                if getattr(lm, 'presence', 1.0) >= 0.3:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    point_color = (0, 0, 255) if is_fall else (255, 255, 255)
                    cv2.circle(output_frame, (cx, cy), 4, point_color, -1)

        # Отрисовка плашки статуса
        if is_fall:
            fall_confidence = max(0.0, min(1.0, (45.0 - body_angle) / 45.0))
        else:
            fall_confidence = 0.0

        status_text = "FALL DETECTED!" if is_fall else "Status: Normal"
        banner_color = (0, 0, 255) if is_fall else (0, 255, 0)

        cv2.putText(
            output_frame,
            f"{status_text} | Angle: {int(body_angle)} deg | Conf: {fall_confidence*100:.0f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            banner_color,
            2,
            cv2.LINE_AA
        )

        return output_frame, is_fall, body_angle, fall_confidence

    def close(self):
        """Освобождение ресурсов MediaPipe."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    def process(self, frame_bgr, timestamp_ms=None):
        """Алиас для совместимости: делегирует в process_frame."""
        return self.process_frame(frame_bgr, timestamp_ms)
