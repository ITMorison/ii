import os
import sys
import time
import numpy as np
import cv2
import traceback

DANGEROUS_ACTIONS = {
    "punch", "punches", "kick", "kicks", "pushing", "push",
    "slapping", "hit", "hitting", "strike", "striking",
    "grabbing", "grab", "shooting", "shoot",
    "breaking", "break", "smashing", "smash",
    "fighting", "fight", "scuffle", "brawl",
    "fire", "firefighters", "flames", "burning",
    "accident", "crash", "car accident",
    "running", "fleeing", "escape", "panic",
}

ACTION_RU_NAMES = {
    "punch": "Удар кулаком",
    "punches": "Много ударов",
    "kick": "Удар ногой",
    "kicks": "Много ударов ногой",
    "pushing": "Толчок",
    "push": "Толчок",
    "slapping": "Хлёсткий удар",
    "hit": "Удар",
    "hitting": "Много ударов",
    "strike": "Удар",
    "striking": "Много ударов",
    "grabbing": "Хватание",
    "grab": "Хватание",
    "shooting": "Выстрел",
    "shoot": "Выстрел",
    "breaking": "Разрушение",
    "break": "Разрушение",
    "smashing": "Разбивание",
    "smash": "Разбивание",
    "fighting": "Драка",
    "fight": "Драка",
    "scuffle": "Схватка",
    "brawl": "Толпа дерётся",
    "fire": "Пожар",
    "firefighters": "Пожарные вызваны",
    "flames": "Пламя",
    "burning": "Горение",
    "accident": "Авария",
    "crash": "Крушение",
    "car accident": "ДТП",
    "running": "Бег",
    "fleeing": "Панический бег",
    "escape": "Бегство",
    "panic": "Паника",
}


class ActionModel:
    """
    Temporal action recognition using mmaction2 TSN (when available)
    or as a bridge to CLIP for frame-level classification.
    Tries mmaction2 first, falls back gracefully when mmcv is absent.
    """

    def __init__(self, device: str = "cpu"):
        self.enabled = False
        self.model = None
        self.device = device
        self.clip_length = 16
        self.action_labels = []
        self._last_prediction_time = 0
        self._prediction_interval = 1.5

        mmaction_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mmaction2-main",
        )
        if not os.path.exists(mmaction_path):
            print("[ActionModel] mmaction2-main directory not found, skipping.")
            return

        try:
            sys.path.insert(0, mmaction_path)
            from mmaction.apis import init_recognizer
            from mmengine import Config

            import torch
            cfg_path = os.path.join(
                mmaction_path,
                "configs",
                "recognition",
                "tsn",
                "tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
            )
            if not os.path.exists(cfg_path):
                print(
                    f"[ActionModel] Config not found at {cfg_path}, "
                    "action recognition unavailable."
                )
                return

            checkpoint = (
                "https://download.openmmlab.com/mmaction/recognition/tsn/"
                "tsn_r50_1x1x3_100e_kinetics400_rgb/"
                "tsn_r50_1x1x3_100e_kinetics400_rgb_20201105-a8e7d758.pth"
            )

            self.model = init_recognizer(cfg_path, checkpoint, device=device)
            self.enabled = True
            self.action_labels = list(
                self.model.cfg.dataset_meta.get("labels", [])
            )
            print(
                f"[ActionModel] mmaction2 TSN initialized, "
                f"{len(self.action_labels)} actions"
            )

        except Exception as e:
            print(f"[ActionModel] mmaction2 unavailable ({e}), using CLIP bridge")
            self.enabled = False
            self.model = None

    def predict_video(self, frames: list, label: str = "") -> dict:
        if not self.enabled or not frames:
            return {"label": "", "confidence": 0.0}

        now = time.time()
        if now - self._last_prediction_time < self._prediction_interval:
            return {"label": "", "confidence": 0.0}

        try:
            import torch
            with torch.no_grad():
                results = self.model.predict(
                    frames, label=label, return_dataloader=False
                )
            self._last_prediction_time = now

            predicted_label = results.get("pred_label", "")
            score = float(results.get("pred_score", 0.0))

            if isinstance(predicted_label, int) and 0 <= predicted_label < len(
                self.action_labels
            ):
                predicted_label = self.action_labels[predicted_label]

            return {"label": str(predicted_label), "confidence": score}

        except Exception as e:
            print(f"[ActionModel] Prediction error: {e}")
            return {"label": "", "confidence": 0.0}

    def is_dangerous(self, label: str) -> bool:
        label_lower = label.lower()
        return any(
            keyword in label_lower for keyword in DANGEROUS_ACTIONS
        )