import os
import random
import time as t
import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
from typing import Callable
from .classes import SparsifyResult
from shapely.geometry import Polygon
from data.logs.logging import setup_logger
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters


DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ORIGINAL_FOLDER = os.path.join(DATA_FOLDER, 'original')
INPUT_GRAPH_FOLDER = os.path.join(DATA_FOLDER, 'input_graph')
INPUT_IMPUTATION_FOLDER = os.path.join(DATA_FOLDER, 'input_imputation')
INPUT_IMPUTATION_ORIGINAL_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'original')
INPUT_ALL_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'all')
INPUT_ALL_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_ALL_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
INPUT_AREA_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'area')
INPUT_AREA_VALIDATION_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'validation')
INPUT_AREA_TEST_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'test')
SPARCIFY_LOG = 'sparcify_log.txt'
logging = setup_logger(name=SPARCIFY_LOG, log_file=SPARCIFY_LOG)

def get_trajectory_df_from_txt(file_path:str) -> gpd.GeoDataFrame:
    """
    Summary:
        Reads a trajectory txt file, add meta data and returns is as a dataframe with srid 3857
        
    Args:
        file_path (str): file_path to txt file with trajectory

    Returns:
        gpd.GeoDataFrame: trajectory as a dataframe
    """
    try:
        df = pd.read_csv(file_path, header=0)
        
        if (df.empty):
            logging.warning(f'No coordinates to extract for {file_path}')
            print('issue......')
            quit()

        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
        df = df.sort_values(by='timestamp')
        df = add_meta_data(trajectory_df=df)
        return df
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')

def add_meta_data(trajectory_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # add meta data
    gdf_next = trajectory_df.shift(-1)
    trajectory_df, gdf_next = get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
    
    bearing_df = calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
    trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

    trajectory_df['dist'] = get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
    # Calculate the time difference between consecutive points
    time_differences = gdf_next['timestamp'] - trajectory_df['timestamp']
    time_differences = time_differences.fillna(0)

    # Calculate speed for points with subsequent points available
    speeds_mps = trajectory_df['dist'] / time_differences
    speeds_mps.fillna(0, inplace=True)
    
    trajectory_df['speed_mps'] = speeds_mps
    trajectory_df['speed_knots'] = trajectory_df['speed_mps'] * 1.943844
    
    return trajectory_df

def sparcify_trajectories_randomly_using_threshold(filepath:str, folderpath: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_points:int = 0
        number_of_points:int = 0
                                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, f'random_{threshold}'.replace('.', '_'))
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        
        trajectory_df['group'] = 0
        
        if boundary_box:
            gdf_bbox = gpd.GeoDataFrame([1], geometry=[boundary_box], crs="EPSG:4326").geometry.iloc[0] 
            trajectory_df['within_boundary_box'] = trajectory_df.within(gdf_bbox)               
            change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(1)
            trajectory_df['group'] = change_detected.cumsum()
            
            # Find the largest group within the boundary box
            group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
            valid_groups = group_sizes[group_sizes >= 2]

            if not valid_groups.empty:
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]

                # Update total number of points
                number_of_points = len(trajectory_filtered)
            else:
                # No valid groups within the boundary box
                return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered = trajectory_df
            number_of_points = len(trajectory_filtered)
        
        if number_of_points < 2:
            return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)        
           
        # Mark ~80% rows for removal
        rows_to_remove = np.random.choice([True, False], size=len(trajectory_filtered), p=[threshold, 1-threshold])
        
        # Ensure first and last isn't removed
        rows_to_remove[0] = rows_to_remove[-1] = False
        
        # Convert rows_to_remove to a Pandas Series
        rows_to_remove_series = pd.Series(rows_to_remove, index=trajectory_filtered.index)
        
        # Reindex rows_to_remove to match trajectory_filtered's index
        rows_to_remove_series = rows_to_remove_series.reindex_like(trajectory_filtered)
        
        # Sparsify
        sparse_trajectory = trajectory_filtered[~rows_to_remove]
        sparse_trajectory.reset_index(drop=True, inplace=True)
        
        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        sparse_trajectory[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered) - len(sparse_trajectory)    
        
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
        quit()
    
def sparcify_trajectories_realisticly(filepath:str, folderpath: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:    
        os_split = '/' if '/' in filepath else '\\'
        reduced_points:int = 0
        number_of_points:int = 0
        
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, 'realistic')
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        trajectory_df['group'] = 0
        
        if boundary_box:
            gdf_bbox = gpd.GeoDataFrame([1], geometry=[boundary_box], crs="EPSG:4326").geometry.iloc[0] 
            trajectory_df['within_boundary_box'] = trajectory_df.within(gdf_bbox)            
            change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(1)
            trajectory_df['group'] = change_detected.cumsum()
            
            # Find the largest group within the boundary box
            group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
            valid_groups = group_sizes[group_sizes >= 2]

            if not valid_groups.empty:
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered_df = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]

                # Update total number of points
                number_of_points = len(trajectory_filtered_df)
            else:
                # No valid groups within the boundary box
                return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            number_of_points = len(trajectory_filtered_df)
            if number_of_points < 2:
                return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)        
        
        if number_of_points < 2:
            return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)        
        
        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        if number_of_points > 2:
            # Convert relevant columns to numpy arrays for faster access
            timestamps = trajectory_filtered_df['timestamp'].to_numpy()
            speed_knots = trajectory_filtered_df['speed_knots'].to_numpy()
            navigational_status = trajectory_filtered_df['navigational_status'].to_numpy()
            cogs = trajectory_filtered_df['cog'].to_numpy()
        
            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            stationed_navigational_status = ['anchored', 'moored']
            
            #first point is always kept
            last_kept_index = 0

            # Loop over points starting from the second one
            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                navigation_last_kept = navigational_status[last_kept_index]
                cog_last_kept = cogs[last_kept_index]
                
                speed_curr = speed_knots[i]
                navigation_curr = navigational_status[i]
                cog_curr = cogs[i]
                    
                keep_conditions = (navigation_last_kept in stationed_navigational_status and navigation_curr in stationed_navigational_status and time_diff > 180) or \
                                (navigation_last_kept in stationed_navigational_status and time_diff > 180) or \
                                (0 <= speed_last_kept <= 14 and 0 <= speed_curr <= 14 and time_diff > 10) or \
                                (0 <= speed_last_kept <= 14 and 0 <= speed_curr <= 14 and cog_last_kept != cog_curr and time_diff > 3.33) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (0 <= speed_last_kept <= 14 and cog_last_kept != cog_curr and time_diff > 3.33 and time_diff > 3.33) or \
                                (0 <= speed_last_kept <= 14 and time_diff > 10) or \
                                (14 < speed_last_kept <= 23 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (14 < speed_last_kept <= 23 and time_diff > 6) or \
                                (speed_last_kept > 23 and time_diff > 2)

                
                # If the condition is false, mark the current point to be kept
                if keep_conditions:
                    keep[i] = True
                    last_kept_index = i

            keep[0] = keep[-1] = True # keep first and last

            sparse_trajectory_df = trajectory_filtered_df[keep]
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')                
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_filepath, sep=',', index=True, header=True, mode='w')     
    
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')    
        quit()

def sparcify_realisticly_strict_trajectories(filepath:str, folderpath: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_points:int = 0
        number_of_points:int = 0
  

        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, 'realistic_strict')
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        
        trajectory_df['group'] = 0
        
        if boundary_box:
            gdf_bbox = gpd.GeoDataFrame([1], geometry=[boundary_box], crs="EPSG:4326").geometry.iloc[0] 
            trajectory_df['within_boundary_box'] = trajectory_df.within(gdf_bbox)                   
            change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(1)
            trajectory_df['group'] = change_detected.cumsum()
            
            # Find the largest group within the boundary box
            group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
            valid_groups = group_sizes[group_sizes >= 2]

            if not valid_groups.empty:
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered_df = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]

                # Update total number of points
                number_of_points = len(trajectory_filtered_df)
            else:
                # No valid groups within the boundary box
                return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            number_of_points = len(trajectory_filtered_df)
        
        if number_of_points < 2:
            return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)        
        
        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        if number_of_points > 2:
            # Convert relevant columns to numpy arrays for faster access
            timestamps = trajectory_filtered_df['timestamp'].to_numpy()
            speed_knots = trajectory_filtered_df['speed_knots'].to_numpy()
            navigational_status = trajectory_filtered_df['navigational_status'].to_numpy()
            cogs = trajectory_filtered_df['cog'].to_numpy()

            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            stationed_navigational_status = ['anchored', 'moored']
            
            # Loop over points starting from the second one
            last_kept_index = 0

            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                navigation_last_kept = navigational_status[last_kept_index]
                cog_last_kept = cogs[last_kept_index]
                
                speed_curr = speed_knots[i]
                navigation_curr = navigational_status[i]
                cog_curr = cogs[i]
                
                keep_conditions = (navigation_last_kept in stationed_navigational_status and navigation_curr in stationed_navigational_status and time_diff > 180) or \
                                (0 <= speed_last_kept <= 14 and 0 <= speed_curr <= 14 and time_diff > 10) or \
                                (0 <= speed_last_kept <= 14 and 0 <= speed_curr <= 14 and cog_last_kept != cog_curr and time_diff > 3.33) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and cog_last_kept != cog_curr and time_diff > 2) 

                # If the condition is false, mark the current point to be kept
                if keep_conditions:
                    keep[i] = True
                    last_kept_index = i
            
            keep[0] = keep[-1] = True # keep first and last

            sparse_trajectory_df = trajectory_filtered_df[keep]
            
            # Drop the rows identified by the indices in the list            
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')        
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            if number_of_points == 2:
                trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_filepath, sep=',', index=True, header=True, mode='w')     
                
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        print(e)
        quit()

def sparcify_large_time_gap_with_threshold_percentage(filepath:str, folderpath: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_points = 0
        number_of_points = 0
          
        trajectory_df = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, f'large_time_gap_{threshold}'.replace('.', '_'))
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)

        trajectory_df['group'] = 0
        
        if boundary_box:
            gdf_bbox = gpd.GeoDataFrame([1], geometry=[boundary_box], crs="EPSG:4326").geometry.iloc[0] 
            trajectory_df['within_boundary_box'] = trajectory_df.within(gdf_bbox)            
            change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(1)
            trajectory_df['group'] = change_detected.cumsum()
            
            # Find the largest group within the boundary box
            group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
            valid_groups = group_sizes[group_sizes >= 2]

            if not valid_groups.empty:
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]

                # Update total number of points
                number_of_points = len(trajectory_filtered)
            else:
                # No valid groups within the boundary box
                return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered = trajectory_df
            number_of_points = len(trajectory_filtered)
            
        if number_of_points < 2:
            return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=False)        
        
        # Step 1: Select a random row, excluding the first and last rows
        # Slice the DataFrame to exclude the first and last rows
        trajectory_df_sliced = trajectory_filtered.iloc[1:-1]

        # Sample from the sliced DataFrame and ensure that start index is chosen such that removing removal_percentage is possible
        random_row_index = int(len(trajectory_df_sliced) * (1-threshold)) + 1

        # Step 2: Determine the range of rows to remove
        # Calculate the number of rows to remove as threshold_percentage% of the total number of rows
        # Ensure it does not exceed the number of rows after the random row
        num_rows_to_remove = int(len(trajectory_filtered) * threshold)
        start_index = max(1, random.randint(1, random_row_index))            
        end_index = min(start_index + num_rows_to_remove, len(trajectory_filtered) - 1)

        # Step 3: Remove the consecutive rows
        rows_to_remove = np.zeros(len(trajectory_filtered), dtype=bool)
        rows_to_remove[start_index:end_index] = True

        # Apply the sparsification
        sparse_trajectory_df = trajectory_filtered[~rows_to_remove]
        sparse_trajectory_df.reset_index(drop=True, inplace=True)

        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        # Save the sparsified trajectory as before
        sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered) - len(sparse_trajectory_df)
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
        quit()