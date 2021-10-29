import logging


def get_logger(name, logger_name=None, level=logging.DEBUG):
    """Get logger."""
    if logger_name is None:
        logger_name = name
    log_file = name + '.log'
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log
