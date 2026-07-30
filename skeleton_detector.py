import os
import sys
import numpy as np

class SkeletonDetector:
    def __init__(self, model_folder: str = None):
        self.enabled = False
        self.opWrapper = None
        
        if model_folder is None:
            model_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'openpose-master', 'models'
            )
        
        try:
            dir_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'openpose-master', 'python', 'openpose', 'Release'
            )
            if os.path.exists(dir_path):
                sys.path.append(dir_path)
            bin_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'openpose-master', 'x64', 'Release'
            )
            if os.path.exists(bin_path):
                os.environ['PATH'] = os.environ.get('PATH', '') + ';' + bin_path
            
            from openpose import pyopenpose as op
            self.op = op
            
            params = {"model_folder": model_folder}
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.enabled = True
        except Exception:
            self.enabled = False
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None:
            return frame
        
        datum = self.op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([datum])
        return datum.cvOutputData
    
    def get_keypoints(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None:
            return None
        
        datum = self.op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([datum])
        return datum.poseKeypoints