import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters

STATS_FOLDER = os.path.dirname(__file__)
STATISTIC_CLEANSING_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing.ndjson')
STATISTIC_CLEANSING_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing.csv')

INPUT_GRAPH_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_graph')
STATISTIC_INPUT_GRAPH_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_graph.ndjson')
STATISTIC_INPUT_GRAPH_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_graph.csv')
VESSELS_INPUT_GRAPH_CSV_FILE = os.path.join(os.path.dirname(__file__), 'vessels_after_cleansing.csv')

INPUT_IMPUTATION_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_imputation')
STATISTIC_INPUT_IMPUTATION_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_imputation.ndjson')
STATISTIC_INPUT_IMPUTATION_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_input_imputation.csv')
VESSELS_INPUT_IMPUTATION_CSV_FILE = os.path.join(os.path.dirname(__file__), 'vessels_after_cleansing.csv')

STATISTIC_AFTER_CLEANSING_TOTAL_JSON_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing_total.ndjson')
STATISTIC_AFTER_TOTAL_CSV_FILE = os.path.join(os.path.dirname(__file__), 'stats_after_cleansing_total.csv')
VESSEL_CSV_TOTAL = os.path.join(os.path.dirname(__file__), 'vessels_after_cleansing_total.csv')

class CleansingStatistics:
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
        self = CleansingStatistics()
            
    @staticmethod
    def instantiate_new() -> 'CleansingStatistics':
        '''Instantiates a new Statistics object.'''
        return CleansingStatistics()

def calculate_cleansing_statistics(df, column_name):
    '''Calculate required statistics for a given column.'''
    data = df[column_name]
    #If the column contains lists (implying dtype is object), concatenate into a single series
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

class Trajectory_Statistics:
    def __init__(self):
        self.filepath: str = ''
        self.position_reports: int = 0
        self.distance_travelled_m: float = 0
        self.vessel_type: str = ''

    def to_dict(self) -> Dict[str, Any]:
        '''Converts the class attributes to a dictionary.'''
        return {
            'filepath': self.filepath,
            'position_reports': self.position_reports,
            'distance_travelled_m': self.distance_travelled_m,
            'vessel_type': self.vessel_type,
        }

    def add_to_file(self, filepath:str) -> None:
        '''Adds latest parsed csv file'''
        with open(filepath, 'a') as file:
            json.dump(self.to_dict(), file)
            file.write('\n') 
        self = Trajectory_Statistics()

    def calculate_distance_traveled(trajectory_df: pd.DataFrame) -> float:
        gdf_next = trajectory_df.shift(-1)
        trajectory_df, gdf_next = get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
        
        bearing_df = calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
        trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

        trajectory_df['dist'] = get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
        distance_travelled = trajectory_df['dist'].sum()

        return distance_travelled

    def make_statistics_for_input_folder(self, input_folder:str, json_file:str, stats_file:str, vessel_file:str):
        '''Create statistics for the input graph folder.'''
        stats = Trajectory_Statistics()
        path = Path(input_folder)

        print(f"Getting number of files")
        number_of_files = sum(1 for file in path.rglob('*.txt'))
        i = 0

        print(f"Making statistics for {number_of_files} files")
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                df = pd.read_csv(os.path.join(root, file))
                stats.filepath = os.path.join(root, file)
                stats.position_reports = len(df)
                stats.vessel_type = df['ship_type'].iloc[0]
                stats.distance_travelled_m = Trajectory_Statistics.calculate_distance_traveled(df)

                stats.add_to_file(json_file)
                i += 1
                sys.stdout.write(f"\rMade stats for {i}/{number_of_files}")
                sys.stdout.flush()
        
        json_df = pd.read_json(json_file, lines=True)
    
        columns_to_calculate = [
            'position_reports', 'distance_travelled_m'
        ]

        # List of columns to calculate statistics for
        stats = {col: calculate_cleansing_statistics(json_df, col) for col in columns_to_calculate}
        
        total_number_of_trajectories_files = json_df['filepath'].count()

        stats['trajectories'] = {'total': total_number_of_trajectories_files, 'average': 0, 'median': 0, 'max': 0, 'min': 0, 'quantile 25%': 0, 'quantile 50%': 0, 'quantile 75%': 0}
        df_stats = pd.DataFrame.from_dict(stats, orient='index', 
                                        columns=['total', 'average', 'median', 'max', 'min', 'quantile 25%', 'quantile 50%', 'quantile 75%'])
        # Writing DataFrame to CSV
        df_stats.to_csv(stats_file)
        
        # Make file for vessels
        vessel_counts = json_df.groupby('vessel_type').size()

        # Converting to dictionary
        vessel_dict = vessel_counts.to_dict()

        # Adding total count to the dictionary
        vessel_dict['Total'] = json_df['vessel_type'].count()

        df_vessel = pd.DataFrame(vessel_dict, index=[0])

        df_vessel.to_csv(vessel_file)

class Sparse_Statistics:
    def __init__(self):
        self.output_folder: str = ''
        self.threshold: int = 0
        self.vessel_samples: int = 0
        self.reduced: int = 0
        self.total_distance: float = 0
        self.vessel_type: str = ''

    def to_dict(self) -> Dict[str, Any]:
        '''Converts the class attributes to a dictionary.'''
        return {
            'Output Folder': self.output_folder,
            'Threshold': self.threshold,
            'Vessel Samples': self.vessel_samples,
            'Reduced': self.reduced,
            'Total Distance': self.total_distance,
            'Vessel Type': self.vessel_type
        }

    def add_to_file(self, filepath:str) -> None:
        '''Adds latest parsed csv file'''
        with open(filepath, 'a') as file:
            json.dump(self.to_dict(), file)
            file.write('\n') 
        self = Trajectory_Statistics()

    def make_statistics_with_threshold(self, input_file:str):
        columns_to_calculate = ['Vessel Samples', 'Reduced', 'Total Distance']
        df = pd.read_json(input_file, lines=True)

        grouped = df.groupby('Output Folder')

        for outputfolder, group in grouped:
            folder_name = outputfolder.split('/')[-2]
            threshold = outputfolder.split('/')[-1].split('.')[0]
            
            stats = {col: calculate_cleansing_statistics(group, col) for col in columns_to_calculate}
            total_number_of_trajectories_files = group['Output Folder'].count()
            stats['trajectories'] = {'total': total_number_of_trajectories_files, 'average': 0, 'median': 0, 'max': 0, 'min': 0, 'quantile 25%': 0, 'quantile 50%': 0, 'quantile 75%': 0}
            df_stats = pd.DataFrame.from_dict(stats, orient='index', 
                                        columns=['total', 'average', 'median', 'max', 'min', 'quantile 25%', 'quantile 50%', 'quantile 75%'])
            
            filepath = os.path.join(STATS_FOLDER, f'input_imputation')
            filepath = os.path.join(filepath, 'test' if 'test' in outputfolder else 'validation')
            filepath = os.path.join(filepath, 'area' if 'area' in outputfolder else 'all')

            if 'area' in filepath:
                area_name = outputfolder.split('/')[-3]
                filepath = os.path.join(filepath, area_name)

            filepath = os.path.join(filepath, folder_name)
            filepath = os.path.join(filepath, threshold)

            os.makedirs(filepath, exist_ok=True)

            filepath_stats = os.path.join(filepath, f'{folder_name}.csv')
            
            # Writing DataFrame to CSV
            df_stats.to_csv(filepath_stats)
        
            # Make file for vessels
            vessel_counts = group.groupby('Vessel Type').size()

            # Converting to dictionary
            vessel_dict = vessel_counts.to_dict()

            # Adding total count to the dictionary
            vessel_dict['Total'] = group['Vessel Type'].count()

            df_vessel = pd.DataFrame(vessel_dict, index=[0])
            vessel_file = os.path.join(filepath, f'{folder_name}_vessels.csv')

            df_vessel.to_csv(vessel_file)

    def make_statistics_no_threshold(self, input_file:str):
        df = pd.read_json(input_file, lines=True)
        columns_to_calculate = ['Vessel Samples', 'Reduced', 'Total Distance']

        grouped = df.groupby('Output Folder')

        for outputfolder, group in grouped:
            folder_name = outputfolder.split('/')[-1]
            
            stats = {col: calculate_cleansing_statistics(group, col) for col in columns_to_calculate}
            total_number_of_trajectories_files = group['Output Folder'].count()
            stats['trajectories'] = {'total': total_number_of_trajectories_files, 'average': 0, 'median': 0, 'max': 0, 'min': 0, 'quantile 25%': 0, 'quantile 50%': 0, 'quantile 75%': 0}
            df_stats = pd.DataFrame.from_dict(stats, orient='index', 
                                        columns=['total', 'average', 'median', 'max', 'min', 'quantile 25%', 'quantile 50%', 'quantile 75%'])
            
            filepath = os.path.join(STATS_FOLDER, f'input_imputation')
            filepath = os.path.join(filepath, 'test' if 'test' in outputfolder else 'validation')
            filepath = os.path.join(filepath, 'area' if 'area' in outputfolder else 'all')
    
            if 'area' in filepath:
                area_name = outputfolder.split('/')[-2]
                filepath = os.path.join(filepath, area_name)

            filepath = os.path.join(filepath, folder_name)

            os.makedirs(filepath, exist_ok=True)

            filepath_stats = os.path.join(filepath, f'{folder_name}2.csv')
            
            # Writing DataFrame to CSV
            df_stats.to_csv(filepath_stats)
        
            # Make file for vessels
            vessel_counts = group.groupby('Vessel Type').size()

            # Converting to dictionary
            vessel_dict = vessel_counts.to_dict()

            # Adding total count to the dictionary
            vessel_dict['Total'] = group['Vessel Type'].count()

            df_vessel = pd.DataFrame(vessel_dict, index=[0])
            vessel_file = os.path.join(filepath, f'{folder_name}_vessels_stats2.csv')

            df_vessel.to_csv(vessel_file)
