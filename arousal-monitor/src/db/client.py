"""
src/db/client.py
────────────────────────────────────────────────────────────────
Supabase session logger.

Logs one row every N seconds to the `sessions` table.
Gracefully degrades (logs a warning, continues) if:
  - SUPABASE_URL / SUPABASE_KEY are not set in .env
  - Supabase is unreachable
  - The table doesn't exist yet

SQL to create the table (run once in Supabase SQL Editor):
──────────────────────────────────────────────────────────────
CREATE TABLE sessions (
  id                UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id        TEXT        NOT NULL,
  timestamp         TIMESTAMPTZ DEFAULT now(),
  blink_rate        FLOAT,
  gaze_aversion     FLOAT,
  pupil_variability FLOAT,
  head_jitter       FLOAT,
  tension_score     FLOAT
);
──────────────────────────────────────────────────────────────
"""

import os
import uuid
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Usage:
        db = SessionLogger()
        db.log(features, tension_score=42.5)
        rows = db.fetch_session()
    """

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.enabled: bool   = False
        self._client         = None

        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()

        if not url or not key:
            logger.warning(
                "SUPABASE_URL or SUPABASE_KEY missing from .env — "
                "DB logging disabled. Session data will be in-memory only."
            )
            return

        try:
            from supabase import create_client
            self._client = create_client(url, key)
            self.enabled = True
            logger.info(f"Supabase connected · session={self.session_id[:8]}…")
        except Exception as exc:
            logger.warning(f"Supabase init failed ({exc}) — DB logging disabled.")

    # ── Public ────────────────────────────────────────────────
    def log(self, features, tension_score: float) -> bool:
        """
        Insert one row. Returns True on success, False otherwise.
        Silently skips when DB is disabled or features is None.
        """
        if not self.enabled or features is None:
            return False

        row = {
            "session_id":        self.session_id,
            "blink_rate":        round(float(features.blink_rate),        4),
            "gaze_aversion":     round(float(features.gaze_aversion),     4),
            "pupil_variability": round(float(features.pupil_variability), 4),
            "head_jitter":       round(float(features.head_jitter),       4),
            "tension_score":     round(float(tension_score),              2),
        }

        try:
            self._client.table("sessions").insert(row).execute()
            return True
        except Exception as exc:
            logger.warning(f"DB insert failed: {exc}")
            return False

    def fetch_session(self) -> list[dict]:
        """Retrieve all rows for the current session, ordered by timestamp."""
        if not self.enabled:
            return []
        try:
            result = (
                self._client
                .table("sessions")
                .select("*")
                .eq("session_id", self.session_id)
                .order("timestamp")
                .execute()
            )
            return result.data or []
        except Exception as exc:
            logger.warning(f"DB fetch failed: {exc}")
            return []

    def is_connected(self) -> bool:
        return self.enabled