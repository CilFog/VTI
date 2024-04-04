from .statistics import Statistics, STATISTIC_JSON_FILE
import os

# Initialize the global instance
stats = Statistics.instantiate_new(STATISTIC_JSON_FILE)