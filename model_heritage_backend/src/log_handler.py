import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class logHandler:

    @staticmethod
    def error_handler(exc, function_name, optional_info=None):
        logger.error(f"Error raised in function '{function_name}': {exc}")
        if optional_info:
            logger.error(f"Additional info: {optional_info}")
        # Additional error handling logic can be added here

    @staticmethod
    def warning_handler(warning, function_name, optional_info=None):
        logger.warning(f"Warning during function '{function_name}': {warning}")
        if optional_info:
            logger.warning(f"Additional info: {optional_info}")
        # Additional warning handling logic can be added here