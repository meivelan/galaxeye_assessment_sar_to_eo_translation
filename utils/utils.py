import logging
from pathlib import Path
from utils.config import Config


def setup_logging(config: Config, run_log_dir: Path = None):
    """
    Setup logging configuration safely matching the new isolated run system.
    
    Args:
        config (Config): The configuration manager instance.
        run_log_dir (Path, optional): Path to the current run's isolated log directory.
                                      If None, falls back to a global log file.
    """
    if run_log_dir is not None:
        log_file = Path(run_log_dir) / "training.log"
    else:
        # Fallback safety net in case it's called before run_id generation
        base_dir = Path(config["logging"]["local"].get("base_dir", "kaggle/working/runs/experiments"))
        log_file = base_dir / "global_training.log"

    # Ensure targeted log parent folder exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear out any stale handlers to prevent duplicate print statements in environments like Jupyter/Kaggle
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging initialized. Writing logs to: {log_file}")


def init_comet(config: Config):
    """Safe fallback wrapper keeping compatibility intact if Comet is fully stripped."""
    return None


def log_metrics(experiment, metrics: dict, step: int, last: int = None):
    """Fallback local terminal metric logger matching the old signature."""
    if last:
        logging.info(f"Epoch {step}/{last} metrics: {metrics}")
    else:
        logging.info(f"Epoch {step} metrics: {metrics}")