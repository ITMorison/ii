import os
import gc
import cv2
import time
import threading

from utils import RTSPStreamReader, record_and_send_incident
from db_analytics import log_alert
from fall_detector import FallDetector
from action_model import ActionModel, ACTION_RU_NAMES
from skeleton_detector import SkeletonDetector

ALERT_COOLDOWN = 15
PRE_EVENT_SECONDS = 5
POST_EVENT_SECONDS = 10

EVENT_RU_NAMES = {
    "fight": "Драка / Насилие",
    "fire": "Пожар / Дым",
    "bullying": "Буллинг",
    "weapon": "Оружие",
    "fall": "Падение / Потеря сознания",
}
CLIP_INFERENCE_INTERVAL = 3  # Run CLIP every N frames
CLIP_FRAME_WIDTH = 224
CLIP_FRAME_HEIGHT = 224

def get_camera_sources(config, settings):
    cameras = {}
    rtsp_cameras = config.get("rtsp_cameras") or {}
    cameras.update(rtsp_cameras)

    if "webcam_index" in config and config.get("webcam_index") is not None:
        cameras.setdefault("Веб-камера", config["webcam_index"])

    if not cameras:
        legacy_source = settings.get("camera_source")
        if legacy_source is not None:
            cameras["Камера №1"] = legacy_source

    return cameras

class CameraManager:
    def __init__(self, cameras: dict, model, settings: dict, config: dict):
        self.model = model
        self.settings = settings
        self.config = config

        self.bot_token = config.get("telegram_token") or os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = config.get("telegram_chat_id") or os.environ.get("TELEGRAM_CHAT_ID")

        self.event_rules = settings.get("event_rules", {})

        self.readers = {}
        self.statuses = {}
        self.status_lock = threading.Lock()
        self.threads = {}

        self.fall_detectors = {}
        self.action_model = None
        self.skeleton_detector = None
        self.action_enabled = False
        self.clip_buffer = {}

        for name, source in cameras.items():
            if str(source).isdigit():
                source = int(source)

            reader = None
            reader_ok = False
            try:
                reader = RTSPStreamReader(source)
                reader.start()
                reader_ok = True
            except Exception as e:
                print(f"[{name}] ❌ Ошибка создания ридера камеры: {e}")

            self.readers[name] = reader

            if getattr(reader, "connected", False) and reader_ok:
                detector = None
                detector_ok = False
                init_error = "(не создан)"
                try:
                    detector = FallDetector()
                    self.fall_detectors[name] = detector
                    detector_ok = True
                    init_error = getattr(detector, '_init_error', None)
                except Exception as e:
                    print(f"[{name}] ❌ Ошибка инициализации FallDetector: {e}")
                    detector_ok = False
                    init_error = str(e)

                status_text = "Инициализация..."
                if not detector_ok:
                    status_text = f"Детектор не загружен: {init_error}"
                elif init_error:
                    status_text = f"Ошибка загрузки MediaPipe: {init_error}"

                self.statuses[name] = {
                    "text": status_text,
                    "confidence": 0.0,
                    "updated_at": time.time()
                }

                if detector_ok and not init_error:
                    self._start_camera_loop(name)

        print(f"[CAMERA MANAGER] Камер запущено: {len(self.fall_detectors)}")

        self.action_model = ActionModel()
        self.action_enabled = self.action_model.enabled

        self.skeleton_detector = SkeletonDetector()

        if self.action_enabled:
            self.clip_buffer = {}
            t = threading.Thread(target=self._action_loop, daemon=True)
            t.start()
            print("[CAMERA MANAGER] Action detection (mmaction2 TSN) active")

        print(f"[CAMERA MANAGER] Система готова")

    def _start_camera_loop(self, name):
        """Запускает поток камеры с авто-перезапуском при падении."""
        def run_loop():
            while True:
                try:
                    self._camera_loop(name)
                except Exception as e:
                    print(f"[{name}] Поток камеры упал: {e}, перезапуск через 5 сек...")
                time.sleep(5)

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        self.threads[name] = t

    def _action_loop(self):
        """Собирает клипы кадров и запускает mmaction2 TSN для распознавания действий."""
        clip_length = 16
        clip_interval = 2.0
        last_run = 0.0

        while True:
            now = time.time()
            if now - last_run < clip_interval:
                time.sleep(0.5)
                continue

            if not self.action_enabled or self.action_model is None:
                time.sleep(1.0)
                continue

            try:
                active_cameras = [
                    name for name in self.get_camera_names()
                    if self.readers.get(name) is not None
                ]
            except Exception:
                time.sleep(1.0)
                continue

            for name in active_cameras:
                stream = self.readers.get(name)
                if stream is None:
                    continue
                frames = []
                for _ in range(clip_length):
                    f = stream.read()
                    if f is not None:
                        frames.append(f)
                    time.sleep(0.04)

                if len(frames) < 8:
                    continue

                try:
                    result = self.action_model.predict_video(frames)
                    label = result.get("label", "")
                    conf = result.get("confidence", 0.0)

                    if label and conf > 0.3 and self.action_model.is_dangerous(label):
                        now2 = time.time()
                        if now2 - stream.last_alert_time > ALERT_COOLDOWN:
                            stream.last_alert_time = now2
                            ru_name = ACTION_RU_NAMES.get(
                                label.lower(), label
                            )
                            print(
                                f"[{name}] 🚨 АКЦИЯ ОБНАРУЖЕНА: "
                                f"{ru_name} ({conf*100:.0f}%)"
                            )
                            try:
                                log_alert(
                                    name, ru_name, conf
                                )
                            except Exception:
                                pass
                            if self.bot_token and self.chat_id:
                                threading.Thread(
                                    target=record_and_send_incident,
                                    kwargs=dict(
                                        stream=stream,
                                        camera_id=name,
                                        event_name=ru_name,
                                        confidence=conf,
                                        bot_token=self.bot_token,
                                        chat_id=self.chat_id,
                                        pre_seconds=PRE_EVENT_SECONDS,
                                        post_seconds=POST_EVENT_SECONDS,
                                    ),
                                    daemon=True,
                                ).start()
                except Exception as e:
                    print(f"[ActionLoop] Ошибка распознавания ({name}): {e}")

            last_run = time.time()

    def _set_status(self, name, text, confidence):
        with self.status_lock:
            self.statuses[name] = {
                "text": text,
                "confidence": confidence,
                "updated_at": time.time()
            }

    def get_status(self, name):
        with self.status_lock:
            return self.statuses.get(name, {
                "text": "Нет подключения",
                "confidence": 0.0,
                "updated_at": time.time()
            })

    def get_camera_names(self):
        return list(self.readers.keys())

    def get_reader(self, name):
        return self.readers.get(name)

    def _resize_for_clip(self, frame):
        return cv2.resize(frame, (CLIP_FRAME_WIDTH, CLIP_FRAME_HEIGHT))

    def _check_clip_events(self, frame, camera_name, frame_count):
        if self.model is None:
            return []

        if frame_count % CLIP_INFERENCE_INTERVAL != 0:
            return []

        detected = []
        try:
            small_frame = self._resize_for_clip(frame)
            predictions = self.model.predict(small_frame)
        except Exception as e:
            print(f"[{camera_name}] Ошибка CLIP-инференса: {e}")
            return detected

        for event_type, prob in predictions.items():
            if prob <= 0.001:
                continue
            rule = self.event_rules.get(event_type, {})
            threshold = rule.get("threshold", 0.25)
            ru_name = rule.get("ru_name", EVENT_RU_NAMES.get(event_type, event_type))
            if prob >= threshold:
                detected.append((ru_name, event_type, prob))
                print(
                    f"[{camera_name}] CLIP [{event_type}] = {prob*100:.1f}% "
                    f"({'▓' * int(prob * 20)}{'░' * (20 - int(prob * 20))} "
                    f"thresh={threshold*100:.0f}%)"
                )
            else:
                print(
                    f"[{camera_name}] CLIP [{event_type}] = {prob*100:.1f}% "
                    f"(below threshold {threshold*100:.0f}%)"
                )

        return detected

    def _camera_loop(self, name):
        stream = self.readers[name]
        detector = self.fall_detectors.get(name)

        if detector is None or getattr(detector, '_init_error', None) is not None:
            print(f"[{name}] Поток запущен без детектора (ошибка инициализации или MediaPipe недоступен).")
            return

        frame_count = 0
        consecutive_errors = 0

        while True:
            try:
                frame = stream.read()

                if frame is None:
                    consecutive_errors += 1
                    if consecutive_errors % 20 == 1:
                        print(f"[{name}]Кадр не получен (попытка {consecutive_errors})")
                    time.sleep(0.5)
                    continue

                consecutive_errors = 0
                frame_count += 1
                if frame_count % 500 == 0:
                    gc.collect()
                clip_events = self._check_clip_events(frame, name, frame_count)
                try:
                    processed_frame, is_fall, angle, clip_conf = detector.process_frame(frame)
                except Exception as e:
                    print(f"[{name}] Ошибка детектора: {e}")
                    is_fall = False
                    angle = 0.0
                    clip_conf = 0.0
                if self.skeleton_detector is not None:
                    try:
                        skel_angle, skel_conf = self.skeleton_detector.get_fall_angle(frame)
                        if skel_conf > 0.5 and skel_angle < 45.0:
                            is_fall = True
                            clip_conf = max(clip_conf, skel_conf)
                    except Exception:
                        pass
                all_events = clip_events[:]
                if is_fall:
                    fall_ru = "Падение / Потеря сознания"
                    if not any(e[1] == "fall" for e in all_events):
                        all_events.append((fall_ru, "fall", clip_conf))
                if all_events:
                    best_event = max(all_events, key=lambda x: x[2])
                    status_text = f"{best_event[0]} ({best_event[2]*100:.0f}%)"
                    status_conf = best_event[2]
                else:
                    status_text = "Норма"
                    status_conf = 0.0
                self._set_status(name, status_text, status_conf)
                now = time.time()
                for ru_name, event_type, confidence in all_events:
                    rule = self.event_rules.get(event_type, {})
                    threshold = rule.get("threshold", 0.25)
                    if confidence >= threshold and (now - stream.last_alert_time > ALERT_COOLDOWN):
                        stream.last_alert_time = now
                        print(f"[{name}] {ru_name.upper()} ОБНАРУЖЕН! Шанс: {confidence*100:.0f}%")
                        try:
                            log_alert(name, ru_name, confidence)
                        except Exception as e:
                            print(f"[{name}] БД ошибка: {e}")
                        if self.bot_token and self.chat_id:
                            threading.Thread(
                                target=record_and_send_incident,
                                kwargs=dict(
                                    stream=stream,
                                    camera_id=name,
                                    event_name=ru_name,
                                    confidence=confidence,
                                    bot_token=self.bot_token,
                                    chat_id=self.chat_id,
                                    pre_seconds=PRE_EVENT_SECONDS,
                                    post_seconds=POST_EVENT_SECONDS,
                                ),
                                daemon=True,
                            ).start()
                time.sleep(0.3)
            except Exception as e:
                consecutive_errors += 1
                print(f"[{name}] Ошибка в цикле камеры (#{consecutive_errors}): {e}")
                if consecutive_errors > 10:
                    print(f"[{name}] Слишком много ошибок подряд, перезапуск потока...")
                    break