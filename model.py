import warnings
warnings.filterwarnings("ignore")
import os
import sys
import subprocess
import torch
import numpy as np
import cv2
import yaml
from PIL import Image

# --- ФИКС CLIP (авто-установка правильной версии) ---
def ensure_clip():
    try:
        import clip
        return clip
    except ImportError:
        print("CLIP не найден. Устанавливаем правильную версию с GitHub...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/openai/CLIP.git"
        ])
        import clip
        return clip

clip = ensure_clip()

# --- ПУТИ ---
BASE_DIR = os.path.dirname(__file__)
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.yaml")

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

# --- МОДЕЛЬ ---
class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Инициализация CLIP на устройстве: {self.device}")

        # папка для весов
        weights_dir = os.path.join(BASE_DIR, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # загрузка модели
        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=self.device,
            download_root=weights_dir
        )
        self.model.eval()

        self.settings = load_settings()

        self.labels = self.settings.get("labels", [
            "normal situation, safe environment",
            "violence, fight, assault, aggressive attack",
            "fire, smoke, flames, burning",
            "bullying, harassment, intimidation",
            "weapon, gun, knife, armed person",
            "person falling, collapsed, lying unconscious"
        ])

        # кодируем текст
        self.text_tokens = clip.tokenize(self.labels).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        print("Модель готова")

    def predict(self, frame_bgr):
        if frame_bgr is None:
            return {}

        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(self.labels))

            raw_results = {
                self.labels[idx]: float(val)
                for val, idx in zip(values, indices)
            }

            mapped = {
                "fight": 0.0,
                "fire": 0.0,
                "bullying": 0.0,
                "weapon": 0.0,
                "fall": 0.0
            }

            for label_text, prob in raw_results.items():
                l = label_text.lower()

                if "fight" in l or "violence" in l:
                    mapped["fight"] = max(mapped["fight"], prob)

                elif "fire" in l or "smoke" in l:
                    mapped["fire"] = max(mapped["fire"], prob)

                elif "bullying" in l or "harassment" in l:
                    mapped["bullying"] = max(mapped["bullying"], prob)

                elif "weapon" in l or "gun" in l or "knife" in l:
                    mapped["weapon"] = max(mapped["weapon"], prob)

                elif "fall" in l or "collapsed" in l or "lying" in l:
                    mapped["fall"] = max(mapped["fall"], prob)

            return mapped

        except Exception as e:
            print(f"Ошибка инференса: {e}")
            return {}

# --- SINGLETON ---
_model_instance = None

def get_predictor_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = Model()
    return _model_instance