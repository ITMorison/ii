import cv2
import threading
import time
import requests
import yaml
import os
import tempfile
from collections import deque

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

_FFMPEG_BASE_TIMEOUT = 10000000  # 10 seconds in microseconds


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _ffmpeg_options_for_source(src):
    """Возвращает строку FFmpeg-опций, специфичных для схемы URL."""
    if not isinstance(src, str):
        return None
    src_lower = src.lower()
    if src_lower.startswith(("rtsp://", "rtsps://")):
        return f"rtsp_transport;tcp|stimeout;{_FFMPEG_BASE_TIMEOUT}|timeout;{_FFMPEG_BASE_TIMEOUT}"
    if src_lower.startswith(("https://", "http://")):
        return f"ssl_verify;0|timeout;{_FFMPEG_BASE_TIMEOUT}"
    return None


class RTSPStreamReader:
    """
    Читает RTSP / HTTP(S)-поток (HLS/m3u8, RTSP, RTMP) в фоновом потоке
    и хранит кольцевой буфер последних кадров, чтобы при срабатывании
    детекции можно было собрать видео «ДО» события.
    """

    def __init__(self, src=0, buffer_seconds=6, target_fps=15):
        self.src = src
        self.frame = None
        self.grabbed = False
        self.started = False
        self.read_lock = threading.Lock()
        self.last_alert_time = 0
        self.connected = False

        self.target_fps = target_fps
        self.buffer = deque(maxlen=int(buffer_seconds * target_fps * 1.5))
        self.buffer_lock = threading.Lock()

        self.cap = self._open_capture(self.src)
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.connected = True
            self._push_to_buffer(self.frame)
            print(f"[CAMERA] Успешно подключено к источнику: {self.src}")
        else:
            print(f"[CAMERA] Не удалось получить первый кадр от источника: {self.src}")

    def _open_capture(self, src):
        is_network = isinstance(src, str) and src.lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )

        cap = cv2.VideoCapture()
        if is_network:
            opts = _ffmpeg_options_for_source(src)
            if opts:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts

            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            cap.open(src, cv2.CAP_FFMPEG)
        else:
            cap.open(src)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            if is_network:
                print(f"[CAMERA] cv2.VideoCapture не смог открыть источник: {src}")
        return cap

    def _push_to_buffer(self, frame):
        with self.buffer_lock:
            self.buffer.append((time.time(), frame))

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        fail_count = 0
        min_interval = 1.0 / self.target_fps
        last_push = 0.0
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                fail_count += 1
                self.connected = False
                if fail_count % 10 == 1:
                    print(
                        f"[CAMERA] Обрыв/нет кадра от {self.src}, "
                        f"попытка переподключения #{fail_count}..."
                    )
                # Переоткрываем тот же объект вместо release+new,
                # чтобы не оставлять зомби-процессы FFmpeg на Windows.
                self.cap.open(self.src, cv2.CAP_FFMPEG)
                continue

            if not self.connected:
                print(f"[CAMERA] Соединение восстановлено: {self.src}")
            self.connected = True
            fail_count = 0

            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.last_frame_time = time.time()

            now = time.time()
            if now - last_push >= min_interval:
                self._push_to_buffer(frame)
                last_push = now

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_pre_event_frames(self, seconds):
        cutoff = time.time() - seconds
        with self.buffer_lock:
            return [(ts, f.copy()) for ts, f in self.buffer if ts >= cutoff]

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()


def _write_video(frames, fps, out_path):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    for codec_name in ('mp4v', 'avc1'):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                continue
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
            writer.release()
            time.sleep(0.2)
            if os.path.exists(out_path):
                size = os.path.getsize(out_path)
                if size > 0:
                    return True
            os.remove(out_path)
        except Exception:
            continue
    return False


def record_and_send_incident(
    stream,
    camera_id,
    event_name,
    confidence,
    bot_token,
    chat_id,
    pre_seconds=5,
    post_seconds=10,
    fps=15,
):
    tmp_path = None
    try:
        detected_at = time.time()
        print(f"[TG BOT] Инцидент: {event_name} (шанс {confidence*100:.0f}%). Собираю видео...")

        pre_frames = stream.get_pre_event_frames(pre_seconds)
        print(f"[TG BOT] Pre-frames собрано: {len(pre_frames)}")

        post_frames = []
        interval = 1.0 / fps
        end_time = time.time() + post_seconds
        while time.time() < end_time:
            frame = stream.read()
            if frame is not None:
                post_frames.append((time.time(), frame))
            time.sleep(interval)

        all_frames = [f for _, f in pre_frames] + [f for _, f in post_frames]
        if not all_frames:
            print("[TG BOT] Ошибка: не удалось собрать ни одного кадра для видео!")
            return

        tmp_path = os.path.join(tempfile.gettempdir(), f"incident_{int(detected_at)}.mp4")
        ok = _write_video(all_frames, fps, tmp_path)
        if not ok:
            print("[TG BOT] Ошибка: не удалось записать видеофайл!")
            return

        file_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        duration = len(all_frames) / fps
        print(f"[TG BOT] Видео готово: {tmp_path}, размер={file_size/1024/1024:.1f}MB, кадров={len(all_frames)}, ~{duration:.1f}с")

        if file_size > 50 * 1024 * 1024:
            print(f"[TG BOT] Файл слишком большой ({file_size/1024/1024:.1f}MB), ограничение Telegram — 50MB")
            return

        caption = (
            f"⚠️ **ВНИМАНИЕ: ИНЦИДЕНТ!**\n\n"
            f"🎥 Камера: {camera_id}\n"
            f"📌 Событие: {event_name}\n"
            f"📊 Шанс: {confidence*100:.0f}%\n"
            f"🕒 Время: {event_dt}\n"
            f"⏱ Длительность видео: ~{duration:.0f} сек "
            f"({pre_seconds} сек до + {post_seconds} сек после начала события)"
        )

        video_sent = False
        for attempt in range(3):
            try:
                url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
                with open(tmp_path, "rb") as video_file:
                    files = {"video": (os.path.basename(tmp_path), video_file, "video/mp4")}
                    data = {
                        "chat_id": str(chat_id),
                        "caption": caption,
                        "parse_mode": "Markdown",
                    }
                    response = requests.post(url, data=data, files=files, timeout=60)

                res_json = response.json()
                if response.status_code == 200 and res_json.get("ok"):
                    print("[TG BOT] Видео с инцидентом успешно доставлено в Telegram!")
                    video_sent = True
                    break
                desc = res_json.get("description", "unknown") if isinstance(res_json, dict) else str(res_json)
                err_code = res_json.get("error_code", "?") if isinstance(res_json, dict) else "?"
                print(f"[TG BOT] Попытка {attempt+1}: sendVideo error [{err_code}]: {desc}")

                if err_code == 400 and "file" in str(desc).lower() and attempt == 0:
                    print("[TG BOT] sendVideo отклонён — повторяем как документ (нет ограничений по кодеку)...")
                    continue

            except requests.exceptions.RequestException as e:
                print(f"[TG BOT] Попытка {attempt+1}: ошибка сети: {e}")

            if attempt < 2:
                time.sleep(3)

        if not video_sent:
            print("[TG BOT] Видео не удалось доставить как видео — пробуем как документ...")
            doc_sent = False
            for attempt in range(2):
                try:
                    url_doc = f"https://api.telegram.org/bot{bot_token}/sendDocument"
                    with open(tmp_path, "rb") as doc_file:
                        files_doc = {"document": (os.path.basename(tmp_path), doc_file, "video/mp4")}
                        data_doc = {"chat_id": str(chat_id), "caption": caption}
                        r_doc = requests.post(url_doc, data=data_doc, files=files_doc, timeout=60)
                    rj_doc = r_doc.json()
                    if r_doc.status_code == 200 and rj_doc.get("ok"):
                        print("[TG BOT] Инцидент доставлен как документ!")
                        doc_sent = True
                        break
                    print(f"[TG BOT] sendDocument попытка {attempt+1} ошибка: {rj_doc}")
                except requests.exceptions.RequestException as e:
                    print(f"[TG BOT] sendDocument ошибка сети: {e}")
                if attempt < 1:
                    time.sleep(2)

            if not doc_sent:
                print("[TG BOT] Документ тоже не доставлен, отправляю фото-превью...")
                try:
                    frame = stream.read()
                    if frame is not None:
                        _, buf = cv2.imencode('.jpg', frame)
                        cap_short = f"⚠️ {event_name} ({confidence*100:.0f}%) — видео/документ недоступны"
                        files_photo = {'photo': ('alert.jpg', buf.tobytes(), 'image/jpeg')}
                        data_photo = {'chat_id': str(chat_id), 'caption': cap_short, 'parse_mode': 'Markdown'}
                        url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                        r_photo = requests.post(url_photo, data=data_photo, files=files_photo, timeout=10)
                        rj_photo = r_photo.json()
                        if r_photo.status_code == 200 and rj_photo.get("ok"):
                            print("[TG BOT] Фото-превью отправлено в Telegram!")
                        else:
                            print(f"[TG BOT] Фото тоже не доставлено: {rj_photo}")
                except Exception as e:
                    print(f"[TG BOT] Ошибка при отправке фото-превью: {e}")

    except Exception as e:
        print(f"[TG BOT] Исключение при сборке/отправке видео: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def capture_and_send_clip(stream, camera_id, event_name, confidence, bot_token, chat_id):
    try:
        print(f"[TG BOT] Попытка отправки алерта (фото): {event_name} ({confidence*100:.1f}%)")
        frame = stream.read()
        if frame is None:
            print("[TG BOT] Ошибка: Не удалось захватить кадр с камеры!")
            return

        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            print("[TG BOT] Ошибка: Не удалось сжать кадр в JPEG!")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        caption = f"**ВНИМАНИЕ: ИНЦИДЕНТ!**\n\n🎥Камера: {camera_id}\n Событие: {event_name}\nУверенность: {confidence*100:.1f}%"

        files = {'photo': ('alert.jpg', encoded_image.tobytes(), 'image/jpeg')}
        data = {'chat_id': str(chat_id), 'caption': caption, 'parse_mode': 'Markdown'}

        response = requests.post(url, data=data, files=files, timeout=10)
        res_json = response.json()

        if response.status_code == 200 and res_json.get("ok"):
            print("[TG BOT] Уведомление успешно доставлено в Telegram!")
        else:
            print(f"[TG BOT] Ошибка Telegram API: {res_json}")

    except Exception as e:
        print(f"[TG BOT] Исключение при отправке в Telegram: {e}")