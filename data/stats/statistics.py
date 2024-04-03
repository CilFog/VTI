import os
from typing import List, Dict, Any
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STATISTIC_FOLDER = os.path.join(DATA_FOLDER, 'stats')
TRAJECTORY_STATISTIC_FILE = os.path.join(STATISTIC_FOLDER, 'trajectory_creation_stats.txt')

class Statistics:
    def __init__(self):
        self.filepath: List[str] = []
        self.initial_rows: List[int] = []
        self.filtered_rows: List[int] = []
        self.trajectory_counts: List[int] = []
        self.rows_per_trajectory: List[List[int]] = []
        self.rows_per_trajectory_after_split: List[List[int]] = []
        self.trajectory_removed_due_to_draught: List[int] = []
        self.trajectory_counts_after_split: List[int] = []
        self.trajectory_removed_due_to_draught_after_split: List[int] = []
        self.distance_travelled_m_per_trajectory_after_split: List[List[float]] = []  # Assuming lengths are floats

    def to_dict(self) -> Dict[str, Any]:
        """Converts the class attributes to a dictionary."""
        return {
            'filepath': self.filepath,
            'initial_rows': self.initial_rows,
            'filtered_rows': self.filtered_rows,
            'trajectory_counts': self.trajectory_counts,
            'rows_per_trajectory': self.rows_per_trajectory,
            'rows_per_trajectory_after_split': self.rows_per_trajectory_after_split,
            'trajectory_removed_due_to_draught': self.trajectory_removed_due_to_draught,
            'trajectory_counts_after_split': self.trajectory_counts_after_split,
            'trajectory_removed_due_to_draught_after_split': self.trajectory_removed_due_to_draught_after_split,
            'distance_travelled_m_per_trajectory_after_split': self.distance_travelled_m_per_trajectory_after_split,
        }

    def save_to_file(self, file_path: str) -> None:
        """Saves the statistics to a file in JSON format."""
        import json
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    def add_to_file(self, file_path:str) -> None:
        """Adds latest parsed csv file"""
        import json
        with open(file_path, 'a') as file:
            json.dump(self.to_dict(), file, indent=4)
            file.write('\n') 
            
    def remove_latest_entry(self, filepath:str):
        try:
            # Check if the last entry in self.filepath matches the filepath to remove
            if self.filepath and self.filepath[-1] == filepath: 
                index_to_remove = len(self.filepath) -1
                self.filepath.pop()
                            
                if index_to_remove < len(self.initial_rows):
                    self.initial_rows.pop()
                if index_to_remove < len(self.filtered_rows):
                    self.filtered_rows.pop()
                if index_to_remove < len(self.trajectory_counts):
                    self.trajectory_counts.pop()
                if index_to_remove < len(self.rows_per_trajectory):
                    self.rows_per_trajectory.pop()
                if index_to_remove < len(self.rows_per_trajectory_after_split):
                    self.rows_per_trajectory_after_split.pop()
                if index_to_remove < len(self.trajectory_removed_due_to_draught):
                    self.trajectory_removed_due_to_draught.pop()
                if index_to_remove < len(self.trajectory_counts_after_split):
                    self.trajectory_counts_after_split.pop()
                if index_to_remove < len(self.trajectory_removed_due_to_draught_after_split):
                    self.trajectory_removed_due_to_draught_after_split.pop()
                if index_to_remove < len(self.distance_travelled_m_per_trajectory_after_split):
                    self.distance_travelled_m_per_trajectory_after_split.pop()
                    
        except:
            return
    
    @staticmethod
    def load_from_file(file_path: str) -> 'Statistics':
        """Loads the statistics from a file and returns an instance of Statistics."""
        return Statistics()
    
    

def make_trajectory_creation_statistic_file():
    df = pd.read_json(TRAJECTORY_STATISTIC_FILE)
    
    if (df.empty):
        print('No stats found')

    