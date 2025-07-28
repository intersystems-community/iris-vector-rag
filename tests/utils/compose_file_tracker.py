# tests/utils/compose_file_tracker.py

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional


class ComposeFileTracker:
    """Tracks changes to COMPOSE_FILE environment variable across test sessions."""
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize ComposeFileTracker.
        
        Args:
            state_file: Path to state file. If None, uses temporary file.
        """
        self.logger = logging.getLogger(__name__)
        self.state_file = state_file or self._get_default_state_file()
        
    def _get_default_state_file(self) -> str:
        """Get default state file path in temp directory."""
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, "iris_compose_file_state.txt")
    
    def get_current_compose_file(self) -> str:
        """Get current COMPOSE_FILE from environment."""
        return os.getenv("COMPOSE_FILE", "docker-compose.yml")
    
    def get_last_compose_file(self) -> Optional[str]:
        """Get last known compose file from state file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            self.logger.warning(f"Failed to read state file: {e}")
        return None
    
    def save_current_compose_file(self) -> None:
        """Save current compose file to state file."""
        current_file = self.get_current_compose_file()
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                f.write(current_file)
            self.logger.debug(f"Saved compose file state: {current_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save state file: {e}")
    
    def has_compose_file_changed(self) -> bool:
        """Check if COMPOSE_FILE environment variable has changed."""
        current_file = self.get_current_compose_file()
        last_file = self.get_last_compose_file()
        
        if last_file is None:
            # First run, save current state
            self.save_current_compose_file()
            return False
        
        changed = current_file != last_file
        if changed:
            self.logger.info(f"Compose file changed: {last_file} -> {current_file}")
            self.save_current_compose_file()
        
        return changed
    
    def clear_state(self) -> None:
        """Clear the state file."""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                self.logger.debug("Cleared compose file state")
        except Exception as e:
            self.logger.warning(f"Failed to clear state file: {e}")