import os
import sys
import math
import numpy as np
import cv2


class SkeletonDetector:
    def __init__(self, model_folder: str = None):
        self.enabled = False
        self.opWrapper = None
        self.fallback_enabled = False
        self.mp_pose = None
        self.mp_drawing = None
        self.fallback_results = None
        self.skeleton_keypoints = None

        if model_folder is None:
            model_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "openpose-master",
                "models",
            )

        # Try OpenPose first (requires build)
        try:
            dir_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "openpose-master",
                "python",
                "openpose",
                "Release",
            )
            if os.path.exists(dir_path):
                sys.path.append(dir_path)
            bin_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "openpose-master",
                "x64",
                "Release",
            )
            if os.path.exists(bin_path):
                os.environ["PATH"] = (
                    os.environ.get("PATH", "") + ";" + bin_path
                )

            from openpose import pyopenpose as op

            self.op = op
            params = {"model_folder": model_folder}
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.enabled = True
            print("[SkeletonDetector] OpenPose loaded successfully")
        except Exception as e:
            print(f"[SkeletonDetector] OpenPose unavailable ({e}), using MediaPipe fallback")
            self._init_mediapipe_fallback()

    def _init_mediapipe_fallback(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.fallback_enabled = True
            print("[SkeletonDetector] MediaPipe Pose fallback active")
        except Exception as e:
            print(f"[SkeletonDetector] MediaPipe also unavailable: {e}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if self.enabled and frame is not None:
            datum = self.op.Datum()
            datum.cvInputData = frame
            self.opWrapper.emplaceAndPop([datum])
            return datum.cvOutputData

        if self.fallback_enabled and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                results = pose.process(rgb)
                self.fallback_results = results
                annotated = frame.copy()
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=3
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2
                        ),
                    )
                return annotated

        return frame

    def get_keypoints(self, frame: np.ndarray) -> np.ndarray:
        if self.enabled and frame is not None:
            datum = self.op.Datum()
            datum.cvInputData = frame
            self.opWrapper.emplaceAndPop([datum])
            if datum.poseKeypoints is not None:
                self.skeleton_keypoints = datum.poseKeypoints
                return datum.poseKeypoints
            return None

        if self.fallback_enabled and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                results = pose.process(rgb)
                self.fallback_results = results
                if results.pose_landmarks:
                    h, w = frame.shape[:2]
                    kps = np.array(
                        [
                            [lm.x * w, lm.y * h, lm.visibility]
                            for lm in results.pose_landmarks.landmark
                        ]
                    )
                    self.skeleton_keypoints = kps
                    return kps
            return None

        return None

    def get_fall_angle(self, frame: np.ndarray) -> tuple:
        keypoints = self.get_keypoints(frame)
        if keypoints is None or len(keypoints) < 25:
            return 0.0, 0.0

        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]

        shoulder_cx = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_cy = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_cx = (left_hip[0] + right_hip[0]) / 2
        hip_cy = (left_hip[1] + right_hip[1]) / 2

        dx = hip_cx - shoulder_cx
        dy = hip_cy - shoulder_cy

        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        confidence = max(0.0, min(1.0, (45.0 - angle) / 45.0))
        return angle, confidence

    def close(self):
        if self.opWrapper is not None:
            self.opWrapper.stop()
        if self.mp_pose is not None:
            self.mp_pose.close()