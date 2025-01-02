# logging_config.py

import colorlog


def setup_logger(name=None):
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
    }
    colorlog.basicConfig(log_colors=log_colors, format=fmt_string, level=colorlog.INFO)
    logger = colorlog.getLogger(name)
    logger.setLevel(colorlog.INFO)
    return logger

# Root logger to be used throughout the project
logger = setup_logger()
