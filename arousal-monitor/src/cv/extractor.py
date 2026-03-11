"""
src/cv/extractor.py
────────────────────────────────────────────────────────────────
Real-time facial signal extractor using MediaPipe Face Mesh.

Extracts:
  - Blink rate (Eye Aspect Ratio threshold method)
  - Iris/pupil area variability (relative, not absolute)
  - Gaze aversion (iris deviation from eye center)
  - Head pose jitter (landmark displacement between frames)

All signals are relative to the frame geometry — no calibrated
physical units. Normalization happens downstream in TensionEngine.
"""

import mediapipe as mp
import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Data Model ────────────────────────────────────────────────
@dataclass
class FeatureVector:
    blink_rate: float = 0.0        # blinks per minute (rolling estimate)
    gaze_aversion: float = 0.0     # mean iris deviation [0..1 approx]
    pupil_variability: float = 0.0 # coefficient of variation of iris area
    head_jitter: float = 0.0       # mean L2 displacement of pose pts
    timestamp: float = field(default_factory=time.time)


# ── Extractor ─────────────────────────────────────────────────
class FacialExtractor:
    """
    Wraps MediaPipe Face Mesh. Call process_frame() per video frame.
    Returns (FeatureVector | None, annotated_bgr_frame).
    """

    # MediaPipe landmark indices
    # --- eye contour (6 pts used for EAR) ---
    _LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
    _RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

    # --- iris (with refine_landmarks=True) ---
    _LEFT_IRIS  = [474, 475, 476, 477]
    _RIGHT_IRIS = [469, 470, 471, 472]

    # --- stable face pts for head pose ---
    _HEAD_PTS = [1, 9, 57, 130, 287, 359]   # nose tip, chin, jaw corners

    # --- eye inner corners (for gaze reference) ---
    _LEFT_EYE_INNER  = 362   # inner corner
    _LEFT_EYE_OUTER  = 263   # outer corner
    _RIGHT_EYE_INNER = 133
    _RIGHT_EYE_OUTER = 33

    # Blink detection params
    EAR_THRESHOLD      = 0.22
    BLINK_CONSEC_FRAMES = 2

    # Rolling buffer sizes (frames)
    IRIS_HIST_SIZE  = 90
    GAZE_HIST_SIZE  = 30
    JITTER_HIST_SIZE = 30

    def __init__(self, fps: int = 15):
        self.fps = fps

        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,       # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing_utils  = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles

        # State
        self._blink_counter  = 0      # consecutive frames below EAR threshold
        self._blink_total    = 0      # total blinks this session
        self._frame_count    = 0      # total frames processed

        self._iris_hist  = []
        self._gaze_hist  = []
        self._jitter_hist = []
        self._prev_pose  = None

    # ── Public ────────────────────────────────────────────────
    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[Optional[FeatureVector], np.ndarray]:
        """
        Args:
            frame: BGR uint8 numpy array (from OpenCV / av)
        Returns:
            (features, annotated_frame)
            features is None when no face is detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if not results.multi_face_landmarks:
            return None, annotated

        lm = results.multi_face_landmarks[0].landmark
        self._frame_count += 1

        # Draw subtle mesh overlay
        self._drawing_utils.draw_landmarks(
            annotated,
            results.multi_face_landmarks[0],
            self._mp_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._drawing_styles.get_default_face_mesh_contours_style(),
        )

        features = FeatureVector(
            blink_rate        = self._compute_blink_rate(lm),
            gaze_aversion     = self._compute_gaze(lm, w, h),
            pupil_variability = self._compute_pupil_var(lm, w, h),
            head_jitter       = self._compute_head_jitter(lm, w, h),
        )
        return features, annotated

    # ── Private helpers ───────────────────────────────────────
    def _pts(self, lm, indices, w, h) -> np.ndarray:
        return np.array([[lm[i].x * w, lm[i].y * h] for i in indices])

    def _ear(self, lm, indices, w, h) -> float:
        """Eye Aspect Ratio — standard Soukupová & Čech formula."""
        p = self._pts(lm, indices, w, h)
        vertical = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
        horizontal = np.linalg.norm(p[0] - p[3])
        return vertical / (2.0 * horizontal + 1e-7)

    def _compute_blink_rate(self, lm) -> float:
        """Rolling blinks-per-minute estimate."""
        # Use a fake w/h = 1 since EAR is ratio-based
        left_ear  = self._ear(lm, self._LEFT_EYE_EAR,  1, 1)
        right_ear = self._ear(lm, self._RIGHT_EYE_EAR, 1, 1)
        avg_ear   = (left_ear + right_ear) / 2.0

        if avg_ear < self.EAR_THRESHOLD:
            self._blink_counter += 1
        else:
            if self._blink_counter >= self.BLINK_CONSEC_FRAMES:
                self._blink_total += 1
            self._blink_counter = 0

        elapsed_min = max(self._frame_count / self.fps / 60, 1 / 60)
        return self._blink_total / elapsed_min

    def _compute_pupil_var(self, lm, w: int, h: int) -> float:
        """Coefficient of variation of iris bounding-box area (proxy for pupil size)."""
        def iris_area(indices):
            pts = self._pts(lm, indices, w, h)
            dx = np.max(pts[:, 0]) - np.min(pts[:, 0])
            dy = np.max(pts[:, 1]) - np.min(pts[:, 1])
            return dx * dy

        area = (iris_area(self._LEFT_IRIS) + iris_area(self._RIGHT_IRIS)) / 2.0
        self._iris_hist.append(area)
        if len(self._iris_hist) > self.IRIS_HIST_SIZE:
            self._iris_hist.pop(0)

        if len(self._iris_hist) < 5:
            return 0.0
        arr = np.array(self._iris_hist)
        return float(np.std(arr) / (np.mean(arr) + 1e-7))

    def _compute_gaze(self, lm, w: int, h: int) -> float:
        """
        Mean relative iris displacement from eye midpoint.
        High value → iris shifted toward corners (aversion).
        """
        def deviation(iris_idx, inner_idx, outer_idx):
            iris_cx = np.mean([lm[i].x for i in iris_idx])
            mid_x   = (lm[inner_idx].x + lm[outer_idx].x) / 2
            eye_w   = abs(lm[outer_idx].x - lm[inner_idx].x) + 1e-7
            return abs(iris_cx - mid_x) / eye_w

        left_dev  = deviation(self._LEFT_IRIS,  self._LEFT_EYE_INNER,  self._LEFT_EYE_OUTER)
        right_dev = deviation(self._RIGHT_IRIS, self._RIGHT_EYE_INNER, self._RIGHT_EYE_OUTER)
        gaze = (left_dev + right_dev) / 2.0

        self._gaze_hist.append(gaze)
        if len(self._gaze_hist) > self.GAZE_HIST_SIZE:
            self._gaze_hist.pop(0)
        return float(np.mean(self._gaze_hist))

    def _compute_head_jitter(self, lm, w: int, h: int) -> float:
        """Mean frame-to-frame L2 displacement of stable head landmarks."""
        pose = self._pts(lm, self._HEAD_PTS, w, h).flatten()

        if self._prev_pose is not None:
            jitter = float(np.linalg.norm(pose - self._prev_pose))
            self._jitter_hist.append(jitter)
            if len(self._jitter_hist) > self.JITTER_HIST_SIZE:
                self._jitter_hist.pop(0)
        self._prev_pose = pose

        return float(np.mean(self._jitter_hist)) if self._jitter_hist else 0.0