from loguru import logger
import time
import functools


def measure_time(func=None, *, message=None):
    if func is None:
        return lambda f: measure_time(f, message=message)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else float('inf')

        # Use custom message if provided, otherwise use default
        log_message = message if message else "Function execution metrics"

        logger.info(
            f"{log_message} | \n"
            f"Function: '{func.__name__}' | \n"
            f"Time: {processing_time:.4f} sec | \n"
            f"FPS: {fps:.2f}"
        )

        return result
    return wrapper
