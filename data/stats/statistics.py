import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

STATISTIC_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats.ndjson')
STATISTIC_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_final.csv')

class Statistics:
    def __init__(self):
        self.filepath: str = ''
        self.initial_rows: int = 0
        self.filtered_rows: int = 0
        self.trajectory_counts: int = 0
        self.rows_per_trajectory: List[int] = []
        self.rows_per_trajectory_after_split: List[int] = []
        self.trajectory_removed_due_to_draught: int = 0
        self.trajectory_counts_after_split: int = 0
        self.trajectory_removed_due_to_draught_after_split: int = 0
        self.distance_travelled_m_per_trajectory_after_split: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        '''Converts the class attributes to a dictionary.'''
        return {
            'filepath': self.filepath,
            'initial_rows': self.initial_rows,
            'filtered_rows': self.filtered_rows,
            'trajectory_counts': self.trajectory_counts,
            'trajectory_counts_after_split': self.trajectory_counts_after_split,
            'rows_per_trajectory': self.rows_per_trajectory,
            'rows_per_trajectory_after_split': self.rows_per_trajectory_after_split,
            'trajectory_removed_due_to_draught': self.trajectory_removed_due_to_draught,
            'trajectory_removed_due_to_draught_after_split': self.trajectory_removed_due_to_draught_after_split,
            'distance_travelled_m_per_trajectory_after_split': self.distance_travelled_m_per_trajectory_after_split,
        }

    def add_to_file(self, filepath:str) -> None:
        '''Adds latest parsed csv file'''
        with open(filepath, 'a') as file:
            json.dump(self.to_dict(), file)
            file.write('\n') 
        self = Statistics()
            
    @staticmethod
    def instantiate_new(filepath: str) -> 'Statistics':
        '''Loads the statistics from a file and returns an instance of Statistics.'''
        return Statistics()


def calculate_statistics(df, column_name):
    '''Calculate required statistics for a given column.'''
    data = df[column_name]
    # If the column contains lists (implying dtype is object), concatenate into a single series
    if data.apply(lambda x: isinstance(x, list)).any():
        data = pd.Series(np.concatenate(data))
    return {
        'total': data.sum(),
        'average': data.mean(),
        'median': data.median(),
        'max': data.max(),
        'min': data.min(),
        'quantile 25%': data.quantile(0.25),
        'quantile 75%': data.quantile(0.75)
    }

def make_trajectory_creation_statistic_file():
    # Open and read the JSON file
    df = pd.read_json(STATISTIC_JSON_FILE, lines=True)

    # List of columns to calculate statistics for
    columns_to_calculate = [
        'initial_rows', 'filtered_rows', 'trajectory_counts', 
        'trajectory_removed_due_to_draught', 'rows_per_trajectory',
        'trajectory_counts_after_split', 'trajectory_removed_due_to_draught_after_split',
        'rows_per_trajectory_after_split', 'distance_travelled_m_per_trajectory_after_split'
    ]

    # Calculating statistics for each category
    stats = {col: calculate_statistics(df, col) for col in columns_to_calculate}

    df_stats = pd.DataFrame.from_dict(stats, orient='index', 
                                      columns=['total', 'average', 'median', 'max', 'min', 'quantile 25%', 'quantile 50%', 'quantile 75%'])


    # Writing DataFrame to CSV
    df_stats.to_csv(STATISTIC_CSV_FILE)
