"""
Session Management for Medical Predictor Chatbot
Handles: session creation, persistence, cleanup, metadata tracking
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Session storage locations
SESSION_DIR = Path("/tmp/deepsense_sessions")
METADATA_FILE = SESSION_DIR / "session_metadata.json"

# Ensure directories exist
SESSION_DIR.mkdir(exist_ok=True)


class SessionManager:
    """Manages session lifecycle, persistence, and metadata tracking"""

    def __init__(self, session_dir: Path = SESSION_DIR):
        self.session_dir = session_dir
        self.metadata_file = METADATA_FILE
        self._load_all_metadata()

    def _load_all_metadata(self):
        """Load all session metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.all_metadata = json.load(f)
            else:
                self.all_metadata = {}
        except Exception as e:
            logger.warning(f"⚠️  Could not load session metadata: {e}")
            self.all_metadata = {}

    def _save_all_metadata(self):
        """Persist all session metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.all_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving session metadata: {e}")

    def create_session(self, session_id: str, user_name: str = "User") -> Dict[str, Any]:
        """
        Create a new session with metadata

        Args:
            session_id: Unique session identifier
            user_name: User name (extracted from greeting)

        Returns:
            Session metadata dict
        """
        metadata = {
            "session_id": session_id,
            "user_name": user_name,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "features_collected": 0,
            "status": "active",
            "reset_count": 0
        }

        self.all_metadata[session_id] = metadata
        self._save_all_metadata()

        logger.info(f"✅ Created session: {session_id} for {user_name}")
        return metadata

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific session"""
        return self.all_metadata.get(session_id)

    def update_session(self, session_id: str, features_collected: int):
        """Update session metadata with current stats"""
        if session_id in self.all_metadata:
            self.all_metadata[session_id]["last_accessed"] = datetime.now().isoformat()
            self.all_metadata[session_id]["features_collected"] = features_collected
            self._save_all_metadata()

    def reset_session(self, session_id: str) -> bool:
        """
        Reset a session: delete session file and update metadata

        Args:
            session_id: Session to reset

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete session state file
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                logger.info(f"🗑️  Deleted session file: {session_file}")

            # Update metadata
            if session_id in self.all_metadata:
                self.all_metadata[session_id]["last_accessed"] = datetime.now().isoformat()
                self.all_metadata[session_id]["features_collected"] = 0
                self.all_metadata[session_id]["reset_count"] += 1
                self.all_metadata[session_id]["status"] = "reset"
                self._save_all_metadata()

                reset_count = self.all_metadata[session_id]["reset_count"]
                logger.info(f"🔄 Session {session_id} reset (reset count: {reset_count})")

            return True
        except Exception as e:
            logger.error(f"❌ Error resetting session {session_id}: {e}")
            return False

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Clean up session files older than max_age_hours

        Args:
            max_age_hours: Sessions older than this are deleted
        """
        try:
            import time
            current_time = time.time()
            deleted_count = 0

            for session_id, metadata in list(self.all_metadata.items()):
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    file_age_hours = (current_time - session_file.stat().st_mtime) / 3600

                    if file_age_hours > max_age_hours:
                        session_file.unlink()
                        logger.info(f"🗑️  Cleaned up old session: {session_id}")
                        deleted_count += 1

            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old session(s)")

        except Exception as e:
            logger.warning(f"⚠️  Error during session cleanup: {e}")

    def get_all_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return {sid: meta for sid, meta in self.all_metadata.items()
                if meta.get("status") == "active"}

    def list_sessions(self) -> str:
        """Get formatted list of all sessions"""
        if not self.all_metadata:
            return "No active sessions"

        lines = ["Active Sessions:"]
        for sid, meta in self.all_metadata.items():
            lines.append(
                f"  • {meta['session_id']}: {meta['user_name']} "
                f"({meta['features_collected']}/16 features, "
                f"reset {meta['reset_count']} times)"
            )
        return "\n".join(lines)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
