from typing import List, Dict, Any

class Statistics:
    def __init__(self):
        self.filepath: List[str] = []
        self.initial_rows: List[int] = []
        self.filtered_rows: List[int] = []
        self.trajectory_counts: List[int] = []
        self.rows_per_trajectory: List[int] = []
        self.rows_per_trajectory_after_split: List[int] = []
        self.trajectory_removed_due_to_draught: List[int] = []
        self.trajectory_counts_after_split: List[int] = []
        self.trajectory_removed_due_to_draught_after_split: List[int] = []
        self.distance_travelled_m_per_trajectory_after_split: List[float] = []  # Assuming lengths are floats

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
            json_data = json.dump(self.to_dict(), indent=4)
            file.write(json_data + '\n')
            
    def remove_latest_entry(self, filepath:str):
        try:
            if self.filepath[-1] == filepath:
                index_to_remove = self.filepath.pop()                
                
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
        import json
        import os
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                stats = Statistics()
                stats.__dict__.update(data)  # Assuming the keys in the file exactly match the attribute names
                return stats
        else:
            return Statistics()