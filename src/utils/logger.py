"""
Logger module: Structured logging system
"""

from loguru import logger as _logger
import sys
from pathlib import Path


def setup_logger(log_dir: str = "./logs", log_level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    _logger.remove()
    
    # Add stdout handler
    _logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file handler
    log_file = Path(log_dir) / "tgnn_training.log"
    _logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=log_level,
        rotation="500 MB"
    )
    
    return _logger


# Default logger instance
logger = _logger
