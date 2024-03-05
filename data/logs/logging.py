import logging
import os

def setup_logger(log_file):
    # Get the directory containing the log file
    log_dir = os.path.dirname(__file__)
    path = os.path.join(log_dir, log_file)
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # Create a file handler
    file_handler = logging.FileHandler(path, mode='a')  # 'a' stands for append mode
        
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger