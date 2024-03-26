import logging
import time
import os

class UTCFormatter(logging.Formatter):
    """A logging formatter that uses UTC timestamps."""
    converter = time.gmtime

def setup_logger(name, log_file, level=logging.DEBUG):
    """Set up a logger with a specific name and log file, using UTC for timestamps."""
    # Get the directory containing the log file
    log_dir = os.path.dirname(__file__)
    path = os.path.join(log_dir, log_file)
    
    # Create a unique logger for each name
    logger = logging.getLogger(name)
    logger.setLevel(level)  # Set the logging level
    
    # Check if the logger already has handlers attached to it
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(path, mode='a')  # 'a' stands for append mode
        
        # Create a formatter with UTC timestamps and set it to the handler
        formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)
    
    return logger