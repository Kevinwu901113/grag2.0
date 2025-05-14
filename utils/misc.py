import os
import logging

def init_logger(work_dir: str):
    log_path = os.path.join(work_dir, "log.txt")
    logger = logging.getLogger("G-RAG")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)

    return logger
