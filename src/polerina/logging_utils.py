import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Track the current log file path to handle re-initialization with new paths
_current_log_file = None

def setup_logger(
    output_path: Optional[str] = None,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    force_info_only: bool = False,
) -> logging.Logger:
    """
    Configures a logger with both console and file handlers.
    Optimized for Jupyter Notebooks and multiprocessing.
    """
    global _current_log_file

    if force_info_only:
        level = logging.INFO
        file_level = logging.INFO
        console_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    log_format = '%(asctime)s [%(process)d] %(levelname)-5s %(name)s: %(message)s'
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Determine if we need to add/update handlers
    # We always refresh handlers if a new output_path is provided to ensure 
    # logs go to the correct experiment folder.
    new_log_file = Path(output_path) / "experiment.log" if output_path else None
    
    if not logger.handlers or (new_log_file and new_log_file != _current_log_file):
        # Clear existing handlers to prevent duplicates or writing to old files
        if logger.hasHandlers():
            for h in logger.handlers[:]:
                logger.removeHandler(h)
                h.close()

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
            
        # File Handler
        if output_path:
            log_dir = Path(output_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(new_log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            _current_log_file = new_log_file
    
    # Silence verbose third-party loggers
    for lib in ['matplotlib', 'PIL', 'numba', 'joblib', 'graphbench']:
        logging.getLogger(lib).setLevel(logging.INFO)

    # Force flush
    for handler in logger.handlers:
        handler.flush()
        
    return logger
