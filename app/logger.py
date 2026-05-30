import logging
import sys
import os

# Configure the logging format and level
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name):
    logger = logging.getLogger(name)
    # Hide noisy Uvicorn access logs unless specifically asked for
    if name == "uvicorn.access" and log_level_str != "TRACE":
        logger.setLevel(logging.WARNING)
    return logger

# Globally quiet HTTP access logs from httpx/uvicorn
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
