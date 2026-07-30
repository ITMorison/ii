import os
import sys
import numpy as np

class ActionModel:
    def __init__(self, device: str = 'cpu'):
        self.enabled = False
        self.model = None
        self.device = device
        
        try:
            mmaction_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'mmaction2-main'
            )
            if os.path.exists(mmaction_path):
                sys.path.insert(0, os.path.dirname(mmaction_path))
            
            from mmaction.apis import init_recognizer
            from mmengine import Config
            
            config_path = os.path.join(
                mmaction_path, 'configs', 'recognition', 'tsn',
                'tsn_r50_1x1x3_100e_kinetics400_rgb.py'
            )
            
            if os.path.exists(config_path):
                cfg = Config.fromfile(config_path)
                checkpoint = 'https://download.openmmlab.com/mmaction/recognition/tsn/' \
                           'tsn_r50_1x1x3_100e_kinetics400_rgb/' \
                           'tsn_r50_1x1x3_100e_kinetics400_rgb_20201105-a8e7d758.pth'
                
                self.model = init_recognizer(cfg, checkpoint, device=device)
                self.enabled = True
        except Exception:
            self.enabled = False

    def predict_video(self, frames: list, label: str = '') -> dict:
        if not self.enabled:
            return {'label': 'Action recognition not available', 'confidence': 0.0}
        return {'label': label, 'confidence': 0.0}