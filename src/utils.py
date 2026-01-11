import logging
import sys
import time
from functools import wraps

def setup_logging(log_file=None):
    """
    Configure logging format and handlers.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def get_logger(name):
    return logging.getLogger(name)

def timer_decorator(func):
    """
    Decorator to measure execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.getLogger(func.__module__).info(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result
    return wrapper
