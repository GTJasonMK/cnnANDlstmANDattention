"""
Centralized logging utility for warning and debug messages
"""
import logging
import os
import sys
import warnings
from typing import Optional
from pathlib import Path
import threading

# Global logger instance
_global_logger = None
_log_lock = threading.Lock()

def setup_centralized_logging(log_file_path: str = "result.log", 
                            log_dir: Optional[str] = None,
                            level: int = logging.WARNING,  # 改为WARNING等级，只显示warning和error
                            capture_warnings: bool = True) -> logging.Logger:
    """
    Setup centralized logging to redirect all warnings and messages to a single log file
    
    Args:
        log_file_path: Name of the log file
        log_dir: Directory to store the log file (default: current working directory)
        level: Logging level
        capture_warnings: Whether to capture Python warnings
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    with _log_lock:
        if _global_logger is not None:
            return _global_logger
        
        # Determine full log file path
        if log_dir:
            log_path = Path(log_dir) / log_file_path
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_path = Path(log_file_path)
        
        # Create logger
        logger = logging.getLogger('evaluation_pipeline')
        logger.setLevel(level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        
        # Create console handler for errors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Capture warnings
        if capture_warnings:
            # Redirect Python warnings to logging
            logging.captureWarnings(True)
            warnings_logger = logging.getLogger('py.warnings')
            warnings_logger.addHandler(file_handler)
            
            # Custom warning handler
            def warning_handler(message, category, filename, lineno, file=None, line=None):
                logger.warning(f"[{category.__name__}] {filename}:{lineno} - {message}")
            
            warnings.showwarning = warning_handler
        
        _global_logger = logger
        logger.info(f"Centralized logging initialized. Log file: {log_path}")
        return logger

def get_logger() -> Optional[logging.Logger]:
    """Get the global logger instance"""
    return _global_logger

def log_info(message: str):
    """Log an info message"""
    if _global_logger:
        _global_logger.info(message)
    else:
        print(f"[INFO] {message}")

def log_warning(message: str):
    """Log a warning message"""
    if _global_logger:
        _global_logger.warning(message)
    else:
        print(f"[WARNING] {message}")

def log_error(message: str):
    """Log an error message"""
    if _global_logger:
        _global_logger.error(message)
    else:
        print(f"[ERROR] {message}")

def log_debug(message: str):
    """Log a debug message"""
    if _global_logger:
        _global_logger.debug(message)
    else:
        # Only print debug if debug mode is enabled
        if os.environ.get("EVAL_DEBUG", "0") != "0":
            print(f"[DEBUG] {message}")

class LogCapture:
    """Context manager to temporarily capture print statements to log file"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or _global_logger
        self.original_stdout = None
        self.original_stderr = None
    
    def __enter__(self):
        if self.logger:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = LogStream(self.logger, logging.INFO)
            sys.stderr = LogStream(self.logger, logging.ERROR)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger and self.original_stdout and self.original_stderr:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

class LogStream:
    """Stream-like object that redirects writes to a logger"""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer = []
    
    def write(self, text: str):
        if text.strip():
            # Clean up common prefixes
            cleaned = text.strip()
            for prefix in ["[INFO]", "[WARNING]", "[ERROR]", "[DEBUG]"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            
            if cleaned:
                self.logger.log(self.level, cleaned)
    
    def flush(self):
        pass

# Enhanced debug logging with centralized capture
def _dlog_centralized(msg: str):
    """Enhanced debug logging that uses centralized logging if available"""
    debug_enabled = os.environ.get("EVAL_DEBUG", "0") != "0"
    if debug_enabled:
        if _global_logger:
            _global_logger.debug(msg)
        else:
            print(f"[DEBUG] {msg}")

# Replace print functions for warning/error capture
def print_warning(msg: str):
    """Print warning that gets captured by centralized logging"""
    if _global_logger:
        _global_logger.warning(msg)
    else:
        print(f"[WARNING] {msg}")

def print_error(msg: str):
    """Print error that gets captured by centralized logging"""
    if _global_logger:
        _global_logger.error(msg)
    else:
        print(f"[ERROR] {msg}")

def print_info(msg: str):
    """Print info that gets captured by centralized logging"""
    if _global_logger:
        _global_logger.info(msg)
    else:
        print(f"[INFO] {msg}")