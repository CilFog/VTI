from .statistics import Statistics
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STATISTIC_FOLDER = os.path.join(DATA_FOLDER, 'stats')
STATISTIC_FILE = os.path.join(STATISTIC_FOLDER, 'stats.json')



# Initialize the global instance
stats = Statistics.load_from_file(STATISTIC_FILE)