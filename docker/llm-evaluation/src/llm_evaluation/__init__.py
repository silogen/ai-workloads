import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(os.path.join(log_dir, "evaluation_metrics.log")), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
