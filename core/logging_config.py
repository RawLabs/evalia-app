import logging
import os
from logging.handlers import RotatingFileHandler

def configure_evalia_logger():
    logger = logging.getLogger("evalia")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplication
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    log_file = "static/evalia_debug.log"
    # Create static directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger