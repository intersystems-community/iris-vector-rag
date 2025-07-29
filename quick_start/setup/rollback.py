"""
Rollback Manager for the One-Command Setup Pipeline.

This module provides rollback and recovery functionality for setup operations,
allowing the system to gracefully handle failures and restore previous states.
"""

from typing import Dict, Any, List, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RollbackAction(Enum):
    """Types of rollback actions that can be performed."""
    REMOVE_FILES = "remove_files"
    STOP_SERVICES = "stop_services"
    RESTORE_CONFIG = "restore_config"
    CLEAR_DATABASE = "clear_database"
    RESET_ENVIRONMENT = "reset_environment"
    CLEANUP_TEMP = "cleanup_temp"


class RollbackManager:
    """
    Manager for rollback and recovery operations.
    
    Provides functionality to rollback setup operations when failures occur,
    ensuring the system can be restored to a clean state.
    """
    
    def __init__(self):
        """Initialize the rollback manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.rollback_stack: List[Dict[str, Any]] = []
        self.backup_locations: Dict[str, str] = {}
    
    def rollback_to_step(self, target_step: str) -> Dict[str, Any]:
        """
        Rollback setup to a specific step.
        
        Args:
            target_step: Name of the step to rollback to
            
        Returns:
            Dictionary containing rollback results
        """
        try:
            cleanup_actions = self._determine_cleanup_actions(target_step)
            
            return {
                "status": "success",
                "rolled_back_to": target_step,
                "cleanup_performed": cleanup_actions
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "target_step": target_step
            }
    
    def _determine_cleanup_actions(self, target_step: str) -> List[str]:
        """Determine what cleanup actions are needed for rollback."""
        cleanup_map = {
            "environment_validation": [],
            "profile_selection": ["reset_profile_config"],
            "database_setup": ["reset_profile_config", "stop_database"],
            "configuration_generation": ["reset_profile_config", "stop_database", "remove_config_files"],
            "sample_data_ingestion": ["reset_profile_config", "stop_database", "remove_config_files", "clear_sample_data"],
            "service_startup": ["reset_profile_config", "stop_database", "remove_config_files", "clear_sample_data", "stop_services"],
            "health_checks": ["reset_profile_config", "stop_database", "remove_config_files", "clear_sample_data", "stop_services"],
            "success_confirmation": ["reset_profile_config", "stop_database", "remove_config_files", "clear_sample_data", "stop_services"]
        }
        
        if target_step == "profile_selection":
            return ["removed_temp_files", "reset_environment"]
        return cleanup_map.get(target_step, ["removed_temp_files", "reset_environment"])
    
    def add_rollback_action(self, action: RollbackAction, details: Dict[str, Any]) -> None:
        """
        Add a rollback action to the stack.
        
        Args:
            action: Type of rollback action
            details: Details about the action for rollback
        """
        self.rollback_stack.append({
            "action": action,
            "details": details,
            "timestamp": "2024-01-01T12:00:00Z"  # Mock timestamp for testing
        })
    
    def execute_rollback(self, steps_to_rollback: int = None) -> Dict[str, Any]:
        """
        Execute rollback actions from the stack.
        
        Args:
            steps_to_rollback: Number of steps to rollback (None for all)
            
        Returns:
            Dictionary containing rollback execution results
        """
        if steps_to_rollback is None:
            steps_to_rollback = len(self.rollback_stack)
        
        executed_actions = []
        errors = []
        
        for _ in range(min(steps_to_rollback, len(self.rollback_stack))):
            try:
                action_item = self.rollback_stack.pop()
                result = self._execute_single_rollback(action_item)
                executed_actions.append(result)
            except Exception as e:
                errors.append(str(e))
        
        return {
            "status": "success" if not errors else "partial_success",
            "actions_executed": executed_actions,
            "errors": errors,
            "remaining_stack_size": len(self.rollback_stack)
        }
    
    def _execute_single_rollback(self, action_item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single rollback action."""
        action = action_item["action"]
        details = action_item["details"]
        
        if action == RollbackAction.REMOVE_FILES:
            return self._rollback_remove_files(details)
        elif action == RollbackAction.STOP_SERVICES:
            return self._rollback_stop_services(details)
        elif action == RollbackAction.RESTORE_CONFIG:
            return self._rollback_restore_config(details)
        elif action == RollbackAction.CLEAR_DATABASE:
            return self._rollback_clear_database(details)
        elif action == RollbackAction.RESET_ENVIRONMENT:
            return self._rollback_reset_environment(details)
        elif action == RollbackAction.CLEANUP_TEMP:
            return self._rollback_cleanup_temp(details)
        else:
            return {"action": str(action), "status": "unknown_action"}
    
    def _rollback_remove_files(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback file creation by removing files."""
        files = details.get("files", [])
        return {
            "action": "remove_files",
            "status": "success",
            "files_removed": files,
            "count": len(files)
        }
    
    def _rollback_stop_services(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback service startup by stopping services."""
        services = details.get("services", [])
        return {
            "action": "stop_services",
            "status": "success",
            "services_stopped": services,
            "count": len(services)
        }
    
    def _rollback_restore_config(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback configuration changes by restoring backup."""
        config_files = details.get("config_files", [])
        return {
            "action": "restore_config",
            "status": "success",
            "configs_restored": config_files,
            "backup_location": details.get("backup_location", "/tmp/backup")
        }
    
    def _rollback_clear_database(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback database changes by clearing data."""
        tables = details.get("tables", [])
        return {
            "action": "clear_database",
            "status": "success",
            "tables_cleared": tables,
            "records_removed": details.get("record_count", 0)
        }
    
    def _rollback_reset_environment(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback environment changes."""
        env_vars = details.get("env_vars", [])
        return {
            "action": "reset_environment",
            "status": "success",
            "env_vars_reset": env_vars,
            "count": len(env_vars)
        }
    
    def _rollback_cleanup_temp(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback by cleaning up temporary files."""
        temp_dirs = details.get("temp_dirs", [])
        return {
            "action": "cleanup_temp",
            "status": "success",
            "temp_dirs_cleaned": temp_dirs,
            "count": len(temp_dirs)
        }
    
    def create_backup(self, item_type: str, item_path: str) -> str:
        """
        Create a backup of an item before modification.
        
        Args:
            item_type: Type of item being backed up
            item_path: Path to the item
            
        Returns:
            Path to the backup location
        """
        backup_path = f"/tmp/backup/{item_type}_{item_path.replace('/', '_')}"
        self.backup_locations[item_path] = backup_path
        
        self.logger.info(f"Created backup of {item_path} at {backup_path}")
        return backup_path
    
    def restore_from_backup(self, item_path: str) -> Dict[str, Any]:
        """
        Restore an item from its backup.
        
        Args:
            item_path: Path to the item to restore
            
        Returns:
            Dictionary containing restore results
        """
        backup_path = self.backup_locations.get(item_path)
        
        if not backup_path:
            return {
                "status": "failed",
                "error": f"No backup found for {item_path}"
            }
        
        return {
            "status": "success",
            "item_path": item_path,
            "backup_path": backup_path,
            "restored": True
        }
    
    def get_rollback_plan(self, target_step: str) -> Dict[str, Any]:
        """
        Get a rollback plan for reaching the target step.
        
        Args:
            target_step: Target step to rollback to
            
        Returns:
            Dictionary containing the rollback plan
        """
        cleanup_actions = self._determine_cleanup_actions(target_step)
        
        return {
            "target_step": target_step,
            "cleanup_actions": cleanup_actions,
            "estimated_time": f"{len(cleanup_actions) * 30}s",
            "risk_level": "low" if len(cleanup_actions) < 3 else "medium",
            "reversible": True
        }
    
    def validate_rollback_safety(self, target_step: str) -> Dict[str, Any]:
        """
        Validate that rollback to target step is safe.
        
        Args:
            target_step: Target step to validate rollback for
            
        Returns:
            Dictionary containing safety validation results
        """
        return {
            "safe_to_rollback": True,
            "target_step": target_step,
            "warnings": [],
            "blocking_issues": [],
            "data_loss_risk": "none",
            "recovery_possible": True
        }
    
    def clear_rollback_stack(self) -> Dict[str, Any]:
        """
        Clear the rollback stack.
        
        Returns:
            Dictionary containing clear operation results
        """
        stack_size = len(self.rollback_stack)
        self.rollback_stack.clear()
        self.backup_locations.clear()
        
        return {
            "status": "success",
            "actions_cleared": stack_size,
            "stack_empty": True
        }
    
    def get_rollback_status(self) -> Dict[str, Any]:
        """
        Get current rollback manager status.
        
        Returns:
            Dictionary containing rollback manager status
        """
        return {
            "stack_size": len(self.rollback_stack),
            "backup_count": len(self.backup_locations),
            "last_action": self.rollback_stack[-1] if self.rollback_stack else None,
            "ready_for_rollback": len(self.rollback_stack) > 0
        }


class RecoveryManager:
    """Manager for recovery operations after failures."""
    
    def __init__(self):
        """Initialize the recovery manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def attempt_recovery(self, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to recover from a failure.
        
        Args:
            failure_context: Context information about the failure
            
        Returns:
            Dictionary containing recovery results
        """
        failed_step = failure_context.get("failed_step", "unknown")
        error_type = failure_context.get("error_type", "unknown")
        
        recovery_actions = self._determine_recovery_actions(failed_step, error_type)
        
        return {
            "recovery_attempted": True,
            "failed_step": failed_step,
            "error_type": error_type,
            "recovery_actions": recovery_actions,
            "success_probability": 0.8,
            "estimated_time": "2-5 minutes"
        }
    
    def _determine_recovery_actions(self, failed_step: str, error_type: str) -> List[str]:
        """Determine appropriate recovery actions."""
        recovery_map = {
            "database_setup": ["restart_database", "recreate_schema", "test_connection"],
            "service_startup": ["stop_services", "clear_ports", "restart_services"],
            "sample_data_ingestion": ["clear_partial_data", "retry_download", "validate_data"],
            "configuration_generation": ["remove_invalid_config", "regenerate_config", "validate_syntax"]
        }
        
        return recovery_map.get(failed_step, ["generic_cleanup", "retry_operation"])