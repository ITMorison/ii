import os
import sys
import contextlib
from io import StringIO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
    "|analyzeduration;5000000|probesize;5000000"
)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "0"

import cv2
import numpy as np
import streamlit as st
import threading
import time
import requests
import json
import base64
import html
from streamlit_autorefresh import st_autorefresh
from model import Model
from skeleton_detector import SkeletonDetector
from action_model import ActionModel
from utils import get_predictor_model, get_skeleton_detector, get_action_model

st.set_page_config(layout="wide")

st.title("Мониторинг безопасности")
st.write("Инициализация...")

def load_camera_config(filepath='config.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Ошибка: Файл {filepath} не найден.")
        return {}

config = load_camera_config()

TELEGRAM_TOKEN = config.get("telegram_token", "")
TELEGRAM_CHAT_ID = config.get("telegram_chat_id", "")

def send_telegram_message(token, chat_id, message, parse_mode=None):
    if not token or not chat_id:
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return requests.post(url, json=payload, timeout=10)

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        threading.Thread(target=send_telegram_message, args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message, "Markdown")).start()
    except Exception as e:
        print(f"Ошибка Telegram: {e}")

model = None
skeleton_detector = None
action_model = None

try:
    st.write("Загрузка моделей ИИ...")
    model = get_predictor_model()
    st.write("Модель CLIP загружена")
    skeleton_detector = get_skeleton_detector()
    st.write("Детектор скелета загружен")
    action_model = get_action_model()
    st.write("Модель действий загружена")
except Exception as e:
    st.error(f"Ошибка загрузки моделей: {e}")
    st.write(f"Детали: {type(e).__name__}: {e}")

CAMERAS_CONFIG = {}
GRID_COLUMNS = 5
GRID_ROWS = 5
MAX_VISIBLE_CAMERAS = GRID_COLUMNS * GRID_ROWS

try:
    webcam_idx = config.get("webcam_index")
    if webcam_idx is not None:
        CAMERAS_CONFIG["Камера 1 (Веб-камера)"] = webcam_idx
    else:
        CAMERAS_CONFIG["Камера 1 (НЕ НАЙДЕНА)"] = "NOT_FOUND"
    for name, url in config.get("rtsp_cameras", {}).items():
        CAMERAS_CONFIG[f"Камера {name}"] = url
    st.write(f"Конфигурация камер: {len(CAMERAS_CONFIG)} шт.")
except Exception as e:
    st.error(f"Ошибка конфигурации камер: {e}")

def encode_frame(frame):
    try:
        _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("utf-8")
    except Exception as e:
        print(f"Ошибка кодирования кадра: {e}")
        return None

def process_camera_frame(frame_rgb, name):
    if model is None or frame_rgb is None:
        return frame_rgb
    try:
        prediction = model.predict(image=frame_rgb)
        label_text = prediction['label'].title()
        confidence = prediction['confidence']
        label_lower = label_text.lower()
        is_alert = any(k in label_lower for k in ["fight", "fire", "crash", "accident"])
        color = (255, 0, 0) if is_alert else (0, 255, 0)
        cv2.putText(frame_rgb, f"{label_text} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if is_alert and confidence > 0.75:
            alert_key = f"{name}_{label_lower}"
            current_time = time.time()
            if alert_key not in st.session_state.last_alerts or (current_time - st.session_state.last_alerts[alert_key]) > ALERT_COOLDOWN:
                event_ru = "👊 ДРАКА" if "fight" in label_lower else "🔥 ПОЖАР" if "fire" in label_lower else "💥 АВАРИЯ"
                tg_message = f"🚨 *УГРОЗА!*\n📍 `{name}` -> *{event_ru}* ({confidence:.2f})"
                send_telegram_alert(tg_message)
                st.session_state.last_alerts[alert_key] = current_time
            if skeleton_detector and skeleton_detector.enabled and confidence >= 0.5:
                    frame_with_skeleton = skeleton_detector.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    if frame_with_skeleton is not None:
                        frame_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Ошибка ИИ: {e}")
    return frame_rgb

def render_dvr_grid():
    cam_names = list(CAMERAS_CONFIG.keys())
    cells = []
    for idx in range(MAX_VISIBLE_CAMERAS):
        if idx < len(cam_names):
            name = cam_names[idx]
            src = CAMERAS_CONFIG[name]
            stream = st.session_state['streams'].get(name)
        else:
            name = None
            src = None
            stream = None
        title = name if name else f"Слот {idx + 1}"
        status = ""
        image_uri = None
        if name is None:
            cell_class = "empty"
        elif src == "NOT_FOUND" or stream is None:
            status = "Нет соединения"
            cell_class = "no-signal"
        elif not stream.connected:
            status = "Нет соединения"
            cell_class = "no-signal"
        else:
            frame = stream.get_frame()
            if frame is None:
                status = "Нет соединения"
                cell_class = "no-signal"
            else:
                frame_resized = cv2.resize(frame, (640, 360))
                if isinstance(src, int):
                    frame_resized = cv2.flip(frame_resized, 1)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_rgb = process_camera_frame(frame_rgb, name)
                image_uri = encode_frame(frame_rgb)
                cell_class = "live"
        title_html = html.escape(title)
        status_html = html.escape(status)
        if image_uri:
            content = f'<img src="{image_uri}" alt="{title_html}">'
        elif status:
            content = f'<div class="no-signal-text">{status_html}</div>'
        else:
            content = '<div class="empty-signal"></div>'
        cells.append(f'''
            <div class="camera-cell {cell_class}">
                <div class="camera-header">{title_html}</div>
                <div class="camera-content">{content}</div>
            </div>
        ''')
    st.markdown(f'''
        <style>
            .dvr-grid {{
                display: grid;
                grid-template-columns: repeat({GRID_COLUMNS}, minmax(0, 1fr));
                grid-template-rows: repeat({GRID_ROWS}, minmax(140px, 1fr));
                gap: 4px;
                width: 100%;
                background: #050505;
                padding: 4px;
                box-sizing: border-box;
            }}
            .camera-cell {{
                position: relative;
                min-width: 0;
                min-height: 0;
                overflow: hidden;
                background: #000;
                border: 1px solid #2a2a2a;
            }}
            .camera-header {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                z-index: 2;
                padding: 4px 8px;
                background: rgba(0, 0, 0, 0.7);
                color: #fff;
                font-size: 12px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .camera-content {{
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #000;
            }}
            .camera-content img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
            .no-signal-text {{
                color: #ff5555;
                font-weight: 700;
                font-size: 18px;
            }}
            .empty-signal {{
                width: 100%;
                height: 100%;
                background: #050505;
            }}
        </style>
        <div class="dvr-grid">{''.join(cells)}</div>
    ''', unsafe_allow_html=True)

def _open_rtsp_capture(source, is_int=False):
    if is_int or source == "NOT_FOUND" or (not isinstance(source, str)) or (not source.strip().lower().startswith('rtsp://')):
        if is_int:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.3)
                return cap
            cap.release()
            cap = None
        return None

    rtsp_url = source.strip()
    cap = None

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        if cap.isOpened():
            return cap
        cap.release()
        cap = None
    except Exception:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    try:
        rtsp_tcp = rtsp_url + ("&" if "?" in rtsp_url else "?") + "rtsp_transport=tcp"
        cap = cv2.VideoCapture(rtsp_tcp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        if cap.isOpened():
            return cap
        cap.release()
        cap = None
    except Exception:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        cap = None
    except Exception:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
    return None


class RTSPStreamReader(threading.Thread):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.frame = None
        self.running = True
        self.connected = False
        self.daemon = True
        self.cap = None

        if source == "NOT_FOUND" or (
            isinstance(source, str) and not source.strip().lower().startswith("rtsp://")
        ):
            self.running = False
            self.connected = False
            return

    def run(self):
        fail_count = 0
        backoff = 0.5
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False
                fail_count += 1
                if fail_count >= 20:
                    try:
                        if self.cap:
                            self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    fail_count = 0
                    self.cap = _open_rtsp_capture(self.source)
                time.sleep(0.5)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception:
                ret = False
                frame = None

            if ret and frame is not None:
                self.frame = frame
                self.connected = True
                fail_count = 0
                backoff = 0.5
            else:
                fail_count += 1
                self.connected = False
                if fail_count >= 20:
                    try:
                        if self.cap:
                            self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    fail_count = 0
                    self.cap = _open_rtsp_capture(self.source)
                time.sleep(0.1)
                continue
            time.sleep(0.03)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.connected = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

def ensure_streams_started():
    if 'streams' not in st.session_state:
        st.session_state['streams'] = {}
    if 'monitor_running' not in st.session_state:
        st.session_state['monitor_running'] = False
    for name, src in CAMERAS_CONFIG.items():
        if name not in st.session_state['streams']:
            stream = RTSPStreamReader(src)
            st.session_state['streams'][name] = stream
            stream.start()
    st.session_state['monitor_running'] = True

if 'last_alerts' not in st.session_state: st.session_state.last_alerts = {}
if 'streams' not in st.session_state: st.session_state.streams = {}
if 'monitor_running' not in st.session_state: st.session_state.monitor_running = False

try:
    ensure_streams_started()
    st.write("Потоки камер запущены")
except Exception as e:
    st.error(f"Ошибка запуска камер: {e}")
    import traceback
    st.code(traceback.format_exc())

ALERT_COOLDOWN = 60

with st.sidebar:
    st.header("Управление системой")
    st.info(f"📹 Камер настроено: {len(CAMERAS_CONFIG)}")
    st.success("✔️ Мониторинг запущен автоматически")
    st.info(f"🖥️ Сетка интерфейса: {GRID_COLUMNS}×{GRID_ROWS}")
    if len(CAMERAS_CONFIG) > MAX_VISIBLE_CAMERAS:
        st.warning(f"На экране показаны первые {MAX_VISIBLE_CAMERAS} камер из config.json")
    with st.expander("Камер подключено"):
        for name, src in CAMERAS_CONFIG.items():
            st.write(f"**{name}**: `{src}`")
    st.subheader("Тест Telegram бота")
    tg_test_token = st.text_input("Telegram token", value=TELEGRAM_TOKEN, type="password")
    tg_test_chat_id = st.text_input("Telegram chat ID", value=TELEGRAM_CHAT_ID)
    tg_test_message = st.text_area("Сообщение", value="Тест подключения Telegram бота")
    if st.button("Отправить тестовое сообщение", use_container_width=True):
        if not tg_test_token or not tg_test_chat_id:
            st.error("Укажите Telegram token и chat ID")
        else:
            try:
                response = send_telegram_message(tg_test_token, tg_test_chat_id, tg_test_message)
                if response is not None and response.ok:
                    st.success("Тестовое сообщение отправлено")
                elif response is not None:
                    st.error(f"Telegram вернул ошибку: {response.status_code} {response.text}")
                else:
                    st.error("Не удалось отправить сообщение")
            except Exception as e:
                st.error(f"Ошибка отправки Telegram: {e}")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        st.success("✔️ Telegram Bot активен")
    else:
        st.warning("⚠️ Telegram Bot не настроен")
    if skeleton_detector and skeleton_detector.enabled:
        st.success("✔️ OpenPose активен")
    else:
        st.warning("⚠️ OpenPose недоступен")
    if action_model and action_model.enabled:
        st.success("✔️ MMAction2 активен")
    else:
        st.warning("⚠️ MMAction2 недоступен")

if st.session_state.monitor_running:
    st_autorefresh(interval=1000, key="datarefresh")

try:
    render_dvr_grid()
except Exception as e:
    st.error(f"Ошибка отрисовки сетки камер: {e}")
    st.write(f"Детали: {type(e).__name__}: {e}")
    import traceback
    st.code(traceback.format_exc())
