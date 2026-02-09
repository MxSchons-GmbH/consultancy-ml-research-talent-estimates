"""
Execution configuration for ML Headcount Pipeline.

This module contains configuration functions that are not part of the Hamilton DAG.
"""

import logging

logger = logging.getLogger(__name__)

# Global configuration for local vs remote execution
# This will be set by the pipeline initialization
_use_remote = True

def set_execution_mode(use_remote: bool = True) -> None:
    """Set the execution mode for all processor functions.
    
    Args:
        use_remote: If True, use remote execution for expensive operations. If False, use local execution.
    """
    global _use_remote
    _use_remote = use_remote
    logger.info(f"Execution mode set to: {'remote' if use_remote else 'local'}")

def execution_mode_is_remote() -> bool:
    """Get the current execution mode.
    
    Returns:
        True if using remote execution, False if using local execution.
    """
    return _use_remote
