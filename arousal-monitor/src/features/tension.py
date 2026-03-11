"""
src/features/tension.py
────────────────────────────────────────────────────────────────
Tension Index Engine

Pipeline:
  1. Calibration phase (30 s) — records user baseline stats
  2. Z-score normalization per feature using calibration mean/std
  3. Weighted sum → raw tension in approx [-3, 3]
  4. Linear rescale → [0, 100]
  5. Exponential Moving Average (α=0.3) → smoothed output

Arousal bands (non-diagnostic labels):
  [0,  30) → Low Arousal
  [30, 60) → Moderate Arousal
  [60,100] → High Arousal
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.cv.extractor import FeatureVector


# ── Output model ──────────────────────────────────────────────
@dataclass
class TensionState:
    score: float               = 0.0
    band: str                  = "Low Arousal"
    band_color: str            = "#4ade80"
    calibrated: bool           = False
    calibration_progress: float = 0.0   # 0.0 → 1.0


# ── Engine ────────────────────────────────────────────────────
class TensionEngine:
    """
    Stateful engine. Feed FeatureVector objects via update().
    Thread-safe: no external locking needed (single-writer pattern).
    """

    # Feature weights — must sum to 1.0
    WEIGHTS = {
        "blink_rate":        0.25,
        "gaze_aversion":     0.30,
        "pupil_variability": 0.20,
        "head_jitter":       0.25,
    }

    CALIBRATION_SECONDS = 30    # seconds before scoring begins
    EMA_ALPHA           = 0.30  # higher → more reactive, lower → smoother
    HISTORY_SIZE        = 600   # 10 minutes at ~1 Hz

    # Z-score clip to prevent outlier distortion
    Z_CLIP = 3.0

    def __init__(self):
        self._cal_buffer: dict[str, list] = {k: [] for k in self.WEIGHTS}
        self._baseline_mean: dict[str, float] = {}
        self._baseline_std:  dict[str, float] = {}
        self._cal_start:     Optional[float]  = None
        self.calibrated = False

        self._ema         = 50.0   # initial neutral value
        self._history     = deque(maxlen=self.HISTORY_SIZE)
        self._timestamps  = deque(maxlen=self.HISTORY_SIZE)

    # ── Public API ────────────────────────────────────────────
    def start_calibration(self):
        """(Re)start the 30-second calibration window."""
        self._cal_start  = time.time()
        self._cal_buffer = {k: [] for k in self.WEIGHTS}
        self.calibrated  = False

    def update(self, features: Optional[FeatureVector]) -> TensionState:
        """
        Process one feature sample.
        Returns TensionState (score=0 during calibration).
        """
        progress = self._calibration_progress()

        # ── No face detected ────────────────────────────────
        if features is None:
            return TensionState(
                score=round(self._ema, 1),
                calibrated=self.calibrated,
                calibration_progress=progress,
            )

        raw_vals = {
            "blink_rate":        features.blink_rate,
            "gaze_aversion":     features.gaze_aversion,
            "pupil_variability": features.pupil_variability,
            "head_jitter":       features.head_jitter,
        }

        # ── Calibration phase ───────────────────────────────
        if not self.calibrated:
            if self._cal_start is None:
                self.start_calibration()

            for k, v in raw_vals.items():
                self._cal_buffer[k].append(v)

            progress = self._calibration_progress()
            if progress >= 1.0:
                self._finalize_calibration()

            return TensionState(
                score=0.0,
                calibrated=False,
                calibration_progress=progress,
            )

        # ── Scoring phase ───────────────────────────────────
        z_scores = {
            k: np.clip(
                (v - self._baseline_mean[k]) / (self._baseline_std[k] + 1e-7),
                -self.Z_CLIP,
                self.Z_CLIP,
            )
            for k, v in raw_vals.items()
        }

        # Weighted sum ∈ [-Z_CLIP, +Z_CLIP]
        raw = sum(self.WEIGHTS[k] * z_scores[k] for k in self.WEIGHTS)

        # Rescale to [0, 100]
        rescaled = (raw + self.Z_CLIP) / (2 * self.Z_CLIP) * 100.0
        rescaled = float(np.clip(rescaled, 0, 100))

        # EMA smoothing
        self._ema = self.EMA_ALPHA * rescaled + (1 - self.EMA_ALPHA) * self._ema

        self._history.append(self._ema)
        self._timestamps.append(time.time())

        band, color = self._get_band(self._ema)

        return TensionState(
            score=round(self._ema, 1),
            band=band,
            band_color=color,
            calibrated=True,
            calibration_progress=1.0,
        )

    def get_history(self) -> tuple[list[float], list[float]]:
        """Returns (timestamps, tension_scores) lists."""
        return list(self._timestamps), list(self._history)

    def detect_pattern(self) -> Optional[str]:
        """
        Heuristic annotations on recent 10-second window.
        Returns a human-readable string or None.
        """
        if len(self._history) < 10:
            return None
        recent = list(self._history)[-10:]
        trend  = recent[-1] - recent[0]
        avg    = float(np.mean(recent))

        if trend > 15 and avg > 50:
            return "📈 Sustained arousal increase detected"
        if recent[-1] > 70 and recent[-3] < 55:
            return "⚡ Sudden spike detected"
        if trend < -15 and avg < 40:
            return "📉 Return to baseline observed"
        if avg > 65:
            return "🔴 Prolonged high arousal"
        return None

    def session_stats(self) -> dict:
        """Summary stats for the current session."""
        if not self._history:
            return {"peak": 0, "mean": 0, "samples": 0}
        h = list(self._history)
        return {
            "peak":    round(max(h), 1),
            "mean":    round(float(np.mean(h)), 1),
            "samples": len(h),
        }

    # ── Private ───────────────────────────────────────────────
    def _calibration_progress(self) -> float:
        if self._cal_start is None:
            return 0.0
        return min((time.time() - self._cal_start) / self.CALIBRATION_SECONDS, 1.0)

    def _finalize_calibration(self):
        for k in self.WEIGHTS:
            data = self._cal_buffer[k]
            self._baseline_mean[k] = float(np.mean(data))   if data else 0.0
            self._baseline_std[k]  = float(np.std(data))    if data else 1.0
            # Guard against zero std (flat signal during calibration)
            if self._baseline_std[k] < 1e-6:
                self._baseline_std[k] = 1.0
        self.calibrated = True

    @staticmethod
    def _get_band(score: float) -> tuple[str, str]:
        if score < 30:
            return "Low Arousal",      "#4ade80"
        if score < 60:
            return "Moderate Arousal", "#facc15"
        return "High Arousal",         "#f87171"