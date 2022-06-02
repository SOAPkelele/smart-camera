import logging


def setup_logger(log_file_name: str):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    logger.info("Script started")
    return logger
