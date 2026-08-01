import json
import os
import time

from model import get_predictor_model
from utils import load_settings
from camera_manager import CameraManager, get_camera_sources

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    settings = load_settings()
    config = load_config()

    print("Загрузка модели ИИ...")
    model = get_predictor_model()
    print("Модель готова.")

    cameras = get_camera_sources(config, settings)
    if not cameras:
        print("ОШИБКА: не найдено ни одной камеры (config.json -> rtsp_cameras / webcam_index).")
        return

    manager = CameraManager(cameras, model, settings, config)

    try:
        while True:
            time.sleep(5)
            for name in manager.get_camera_names():
                status = manager.get_status(name)
                print(f"[{name}] {status['text']}")
    except KeyboardInterrupt:
        print("Остановка сервиса...")


if __name__ == "__main__":
    main()