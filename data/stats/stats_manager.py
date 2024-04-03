from .statistics import Statistics, DATA_FOLDER, STATISTIC_FOLDER
import os

STATISTIC_FILE = os.path.join(STATISTIC_FOLDER, 'stats.ndjson')

# Initialize the global instance
stats = Statistics.load_from_file(STATISTIC_FILE)