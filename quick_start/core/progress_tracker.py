"""
Progress tracking for the Quick Start setup process.

This module provides progress monitoring and reporting capabilities
for the quick start setup workflow.
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class ProgressUpdate:
    """Progress update information."""
    phase: str
    progress: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]


class ProgressTracker:
    """Tracks and reports progress during setup operations."""
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.current_phase = None
        self.progress = 0.0
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add a progress callback."""
        self.callbacks.append(callback)
    
    def update_progress(
        self, 
        phase: str, 
        progress: float, 
        message: str = "",
        details: Dict[str, Any] = None
    ):
        """Update the current progress."""
        self.current_phase = phase
        self.progress = progress
        
        update = ProgressUpdate(
            phase=phase,
            progress=progress,
            message=message,
            details=details or {}
        )
        
        for callback in self.callbacks:
            callback(update)
    
    def get_current_progress(self) -> ProgressUpdate:
        """Get the current progress state."""
        return ProgressUpdate(
            phase=self.current_phase or "unknown",
            progress=self.progress,
            message="",
            details={}
        )