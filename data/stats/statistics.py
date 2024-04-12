import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any

STATISTIC_CLEANSING_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing.ndjson')
STATISTIC_CLEANSING_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing.csv')
STATISTIC_CLEANSING_VESSELS_CSV_FILE = os.path.join(os.path.dirname(__file__), 'vessels_after_cleansing.csv')
INPUT_GRAPH_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_graph')
STATISTIC_INPUT_GRAPH_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_graph.ndjson')
STATISTIC_INPUT_GRAPH_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_graph.csv')

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
    def instantiate_new() -> 'Statistics':
        '''Instantiates a new Statistics object.'''
        return Statistics()


def calculate_cleansing_statistics(df, column_name):
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

def make_trajectory_cleansing_statistic_file(input_file:str, output_file:str):
    # Open and read the JSON file
    df = pd.read_json(input_file, lines=True)

    # List of columns to calculate statistics for
    columns_to_calculate = [
        'initial_rows', 'filtered_rows', 'trajectory_counts', 
        'trajectory_removed_due_to_draught', 'rows_per_trajectory',
        'trajectory_counts_after_split', 'trajectory_removed_due_to_draught_after_split',
        'rows_per_trajectory_after_split', 'distance_travelled_m_per_trajectory_after_split'
    ]

    # Calculating statistics for each category
    stats = {col: calculate_cleansing_statistics(df, col) for col in columns_to_calculate}

    total_number_of_csv_files = df['filepath'].count()

    stats['csv_files'] = {'total': total_number_of_csv_files, 'average': 0, 'median': 0, 'max': 0, 'min': 0, 'quantile 25%': 0, 'quantile 50%': 0, 'quantile 75%': 0}

    df_stats = pd.DataFrame.from_dict(stats, orient='index', 
                                      columns=['total', 'average', 'median', 'max', 'min', 'quantile 25%', 'quantile 50%', 'quantile 75%'])


    # Writing DataFrame to CSV
    df_stats.to_csv(STATISTIC_CLEANSING_CSV_FILE)

def get_number_of_vessel_trajectories_in_folder(input_folder:str, output_file_str):
    '''Count the number of vessel trajectories in a folder and write the result to a csv file.'''
    vessels = ['Anti-pollution', 'Dredging', 'Passenger', 'Port_tender', 'Towing_long_wide','Cargo', 'Fishing', 'Pilot', 'Tanker', 'Tug', 'Diving', 'Law_enforcement', 'Pleasure', 'Towing']
    vessel_stats = []

    # Dictionary to accumulate file counts
    file_counts = {vessel: 0 for vessel in vessels}

    for root, dirs, files in os.walk(input_folder): 
        last_segment = os.path.basename(root)
        if last_segment in vessels:
                # Sum file counts for the current directory and all its subdirectories
                file_counts[last_segment] += sum(len(files) for _, _, files in os.walk(root))

    # Creating entries for the DataFrame
    for vessel, count in file_counts.items():
        vessel_stats.append({'Vessel type': vessel, 'Number of trajectories': count})
    
    # Handling case where no files are found
    if not vessel_stats:
        raise ValueError('No vessel trajectories found in the folder')

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(vessel_stats)
    df.to_csv(output_file_str)

make_trajectory_cleansing_statistic_file(input_file=STATISTIC_CLEANSING_JSON_FILE, output_file=STATISTIC_CLEANSING_CSV_FILE)
get_number_of_vessel_trajectories_in_folder(input_folder=INPUT_GRAPH_FOLDER, output_file_str=STATISTIC_CLEANSING_VESSELS_CSV_FILE)