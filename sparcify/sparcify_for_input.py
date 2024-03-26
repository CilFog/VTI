import os
import sys
import random
import datetime as dt
import time as t
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Callable
from .classes import SparsifyResult
from shapely.geometry import box, Polygon
from multiprocessing import freeze_support
from data.logs.logging import setup_logger
from concurrent.futures import ProcessPoolExecutor
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters


DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ORIGINAL_FOLDER = os.path.join(DATA_FOLDER, 'original')
INPUT_FOLDER = os.path.join(DATA_FOLDER, 'input')
INPUT_ALL_FOLDER = os.path.join(INPUT_FOLDER, 'all')
INPUT_ALL_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_ALL_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
INPUT_AREA_FOLDER = os.path.join(INPUT_FOLDER, 'area')
INPUT_AREA_VALIDATION_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'validation')
INPUT_AREA_TEST_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'test')
SPARCIFY_LOG = 'sparcify_log.txt'

logging = setup_logger(name=SPARCIFY_LOG, log_file=SPARCIFY_LOG)

def get_trajectory_df_from_txt(file_path:str) -> gpd.GeoDataFrame:
    """
    Summary:
        Reads a trajectory txt file and returns is as a dataframe with srid 3857
        
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
        return df
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')

def filter_original_trajectories(sog_threshold: float):
    removed_ship_type = 0
    removed_sog = 0
    removed_draught = 0
    removed_duplicate = 0
    ship_types = ['fishing', 'tanker', 'tug', 'cargo', 'passenger', 'dredging', 'law enforcement', 'anti-pollution', 'pilot', 'pleasure', 'towing', 'port tender', 'diving', 'towing long/wide', ''] 
    moved = 0
    initial_num = 0
    try:
        initial_time = t.perf_counter()
        logging.info('Began filtering original trajectories')   
        for root, folder, files in os.walk(ORIGINAL_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
                initial_num += 1
                only_null_ship_type = trajectory_df.ship_type.isnull().all() or trajectory_df.ship_type.isna().all()
                
                if only_null_ship_type:
                    removed_ship_type += 1
                    os.remove(file_path)
                    
                    if not any(os.scandir(root)):
                        os.rmdir(root)
                    continue
                
                some_null_ship_type = trajectory_df.ship_type.isnull().any() or trajectory_df.ship_type.isna().any()
                if some_null_ship_type:
                    first_non_null_index = trajectory_df['ship_type'].first_valid_index()
                    trajectory_df = trajectory_df['ship_type'].fillna(trajectory_df.loc[first_non_null_index, 'ship_type'])

                if trajectory_df.iloc[0].ship_type.lower() not in ship_types:
                    removed_ship_type += 1
                    os.remove(file_path)
                    if not any(os.scandir(root)):
                        os.rmdir(root)
                    continue
                
                filtered_df = trajectory_df[trajectory_df['sog'] > sog_threshold].copy()
                
                if len(filtered_df) < 2:
                    removed_sog += 1
                    os.remove(file_path)
                    if not any(os.scandir(root)):
                        os.rmdir(root)
                    continue
                
                only_null_draught = filtered_df.draught.isnull().all() or filtered_df.draught.isna().all()
                
                if only_null_draught:
                    removed_draught += 1
                    os.remove(file_path)
                    if not any(os.scandir(root)):
                        os.rmdir(root)
                    
                    continue
                    
                # Create a boolean mask to identify rows where the next row has different latitude or longitude
                mask = (filtered_df['latitude'] != filtered_df['latitude'].shift(-1)) | (filtered_df['longitude'] != filtered_df['longitude'].shift(-1))

                # Apply the mask to keep only the rows where the next row has different latitude or longitude
                filtered_df = filtered_df[mask]
                
                if len(filtered_df) < 2:
                    removed_duplicate += 1
                    os.remove(file_path)
                    if not any(os.scandir(root)):
                        os.rmdir(root)
                    
                    continue
                
                some_null_draught = filtered_df.draught.isnull().any() or filtered_df.draught.isna().any()
                
                file_name = file.split('/')[-1]
                vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_').replace(' ', '_')
                mmsi = root.split('/')[-1]
                new_folder_path = f'{ORIGINAL_FOLDER}/{vessel_folder}/{mmsi}'
                os.makedirs(new_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
            
                # only sog updated
                if not some_null_draught and not len(filtered_df) == len(trajectory_df):
                    datetime_object = dt.datetime.fromtimestamp(filtered_df.iloc[0].timestamp, tz=dt.timezone.utc)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-')            
                    file_name = f'{mmsi}_{str_datetime}.txt'
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                
                # nothing to update
                elif not some_null_draught and len(filtered_df) == len(trajectory_df):
                    file_name = file_name.replace('/', '-').replace(' ', '_').replace(':', '-') 
                    
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                
                # only druaght updated
                elif some_null_draught and len(filtered_df) == len(trajectory_df):
                    max_draught = filtered_df['draught'].max()
                    filtered_df['draught'] = filtered_df['draught'].fillna(max_draught)
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                    
                # both updated
                else:
                    max_draught = filtered_df['draught'].max()
                    filtered_df['draught'] = filtered_df['draught'].fillna(max_draught) # update draught
                    
                    # fix name, and remove old file
                    datetime_object = dt.datetime.fromtimestamp(filtered_df.iloc[0].timestamp, tz=dt.timezone.utc)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-')             
                    file_name = f'{mmsi}_{str_datetime}.txt'
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

                
                
                #remove file
                os.remove(file_path)
                if not any(os.scandir(root)):
                    os.rmdir(root)
                
                moved += 1
    
    except Exception as e:
        logging.error(f'Failed with {repr(e)}')
        quit()
    
    removed = removed_draught + removed_ship_type + removed_sog + removed_duplicate
    
    finished_time = t.perf_counter() - initial_time
    logging.info(f'\nRemoved_due_to_ship: {removed_ship_type}\nRemoved_due_to_sog: {removed_sog}\nRemoved_due_to_draught: {removed_draught}\nRemoved_due_to_duplicate: {removed_duplicate}\nTotal removed: ({removed}/{initial_num})\nTotal moved to new location: ({moved}/{initial_num})\nElapsed time: {finished_time:0.4f} seconds"')   

def sparcify_trajectories_randomly_using_threshold(file_path:str, folder_path: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        reduced_points:int = 0
        number_of_points:int = 0
                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
        file_name = os.path.basename(file_path)
        vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
        vessel_folder_path = os.path.join(folder_path, vessel_folder)
        new_file_path = os.path.join(vessel_folder_path, file_name)
        
        trajectory_df:gpd.GeoDataFrame = add_meta_data(trajectory_df=trajectory_df)
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

        sparse_trajectory[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered) - len(sparse_trajectory)    
        
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
    
def sparcify_realisticly_trajectories(file_path:str, folder_path: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:    
        reduced_points:int = 0
        number_of_points:int = 0
                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
        file_name = os.path.basename(file_path)
        vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
        vessel_folder_path = os.path.join(folder_path, vessel_folder)
        new_file_path = os.path.join(vessel_folder_path, file_name)
        
        trajectory_df:gpd.GeoDataFrame = add_meta_data(trajectory_df=trajectory_df)
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
            navigation_status = trajectory_filtered_df['navigation_status'].to_numpy()
            cogs = trajectory_filtered_df['navigation_status'].to_numpy()
        
            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            stationed_navigation_status = ['anchored', 'moored']
            
            # Loop over points starting from the second one
            last_kept_index = 0
            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                navigation_last_kept = navigation_status[last_kept_index]
                cog_last_kept = cogs[last_kept_index]
                
                speed_curr = speed_knots[i]
                navigation_curr = navigation_status[i]
                cog_curr = cogs[i]
                    
                keep_conditions = (navigation_last_kept in stationed_navigation_status and navigation_curr in stationed_navigation_status and time_diff > 180) or \
                                (navigation_last_kept in stationed_navigation_status and time_diff > 180) or \
                                (0 < speed_last_kept <= 14 and 0 <= speed_curr <= 14 and time_diff > 10) or \
                                (0 < speed_last_kept <= 14 and 0 <= speed_curr <= 14 and cog_last_kept != cog_curr and time_diff > 3.33) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6) or \
                                (14 < speed_last_kept <= 23 and 14 < speed_curr <= 23 and time_diff > 6 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and time_diff > 2) or \
                                (speed_last_kept > 23 and speed_curr > 23 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (0 < speed_last_kept <= 14 and cog_last_kept != cog_curr and time_diff > 3.33 and time_diff > 3.33) or \
                                (0 < speed_last_kept <= 14 and time_diff > 10) or \
                                (14 < speed_last_kept <= 23 and cog_last_kept != cog_curr and time_diff > 2) or \
                                (14 < speed_last_kept <= 23 and time_diff > 6) or \
                                (speed_last_kept > 23 and time_diff > 2)

                
                # If the condition is false, mark the current point to be kept
                if keep_conditions:
                    keep[i] = True
                    last_kept_index = i
            
            keep[0] = keep[-1] = True # keep first and last

            sparse_trajectory_df = trajectory_filtered_df[keep]
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')                
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_file_path, sep=',', index=True, header=True, mode='w')     
    
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')    

def sparcify_realisticly_strict_trajectories(file_path:str, folder_path: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        reduced_points:int = 0
        number_of_points:int = 0
                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
        file_name = os.path.basename(file_path)
        vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
        vessel_folder_path = os.path.join(folder_path, vessel_folder)
        new_file_path = os.path.join(vessel_folder_path, file_name)
        
        trajectory_df:gpd.GeoDataFrame = add_meta_data(trajectory_df=trajectory_df)
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
            navigation_status = trajectory_filtered_df['navigation_status'].to_numpy()
            cogs = trajectory_filtered_df['navigation_status'].to_numpy()

            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            stationed_navigation_status = ['anchored', 'moored']
            
            # Loop over points starting from the second one
            last_kept_index = 0
            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                navigation_last_kept = navigation_status[last_kept_index]
                cog_last_kept = cogs[last_kept_index]
                
                speed_curr = speed_knots[i]
                navigation_curr = navigation_status[i]
                cog_curr = cogs[i]
                
                keep_conditions = (navigation_last_kept in stationed_navigation_status and navigation_curr in stationed_navigation_status and time_diff > 180) or \
                                (0 < speed_last_kept <= 14 and 0 <= speed_curr <= 14 and time_diff > 10) or \
                                (0 < speed_last_kept <= 14 and 0 <= speed_curr <= 14 and cog_last_kept != cog_curr and time_diff > 3.33) or \
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
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')        
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            if number_of_points == 2:
                trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_file_path, sep=',', index=True, header=True, mode='w')     
                
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')

def sparcify_large_time_gap_with_threshold_percentage(file_path:str, folder_path: str, threshold:float = 0.0, boundary_box:Polygon = None) -> SparsifyResult:
    try:
        reduced_points:int = 0
        number_of_points:int = 0
                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
        file_name = os.path.basename(file_path)
        vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
        vessel_folder_path = os.path.join(folder_path, vessel_folder)
        new_file_path = os.path.join(vessel_folder_path, file_name)
        
        trajectory_df:gpd.GeoDataFrame = add_meta_data(trajectory_df=trajectory_df)
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
        sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered) - len(sparse_trajectory_df)
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')

def get_files_in_range(start_date, end_date, directory):
    """
    Lists files in the specified directory with dates in their names falling between start_date and end_date.
    
    :param start_date: Start date in DD-MM-YYYY format
    :param end_date: End date in DD-MM-YYYY format
    :param directory: Directory to search for files
    :return: List of filenames that fall within the date range
    """
    start_date = dt.datetime.strptime(start_date, '%d-%m-%Y').date()
    end_date = dt.datetime.strptime(end_date, '%d-%m-%Y').date()
    files_in_range = []

    for filename in os.scandir(directory):
        print(filename)
        # Extract the date part from the filename, assuming format like '111219502_01-03-2023_11-45-51'
        parts = filename.split('_')
        if len(parts) < 3:
            logging.error(f'Incorrect nameformat for {filename}')
            quit()  # Not enough parts in the filename
        date_str = parts[1]  # The date part is expected to be in the middle
        print(filename)
        try:
            file_date = dt.datetime.strptime(date_str, '%d_%m_%Y').date()
            if start_date <= file_date <= end_date:
                files_in_range.append(filename)
        except ValueError:
            # If date parsing fails, ignore the file
            pass

    return files_in_range

def sparcify_trajectories_with_action_for_folder(
    str_start_date: str,
    str_end_date: str,
    folder_path: str, 
    action: Callable[[str, float, Polygon], SparsifyResult], 
    threshold: float = 0.0,  # Example default value for threshold
    boundary_box: Polygon = None  # Default None, assuming Polygon is from Shapely or a similar library
):
    initial_time = t.perf_counter()
    total_reduced_points = 0
    total_number_of_points = 0
    total_trajectories = 0

    logging.info(f'Began sparcifying trajectories with {str(action)}')

    print('gettings files')
    # file_paths = get_files_in_range(start_date=str_start_date, end_date=str_end_date, directory=ORIGINAL_FOLDER)
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(ORIGINAL_FOLDER) for file in files]
    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(action, file_paths, [folder_path] * len(file_paths), [threshold] * len(file_paths), [boundary_box] * len(file_paths))        
        for result in results:
            total_reduced_points += result.reduced_points
            total_number_of_points += result.number_of_points
            total_trajectories += 1 if result.trajectory_was_used else 0

    reduced_avg = total_reduced_points/total_trajectories
    finished_time = t.perf_counter() - initial_time
    
    logging.info(f'Reduced on avg. pr trajectory: {reduced_avg} for {total_trajectories} trajectories. Reduced points in total: {total_reduced_points}/{total_number_of_points}. Elapsed time: {finished_time:0.4f} seconds')   

def add_meta_data(trajectory_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # add meta data
    gdf_next = trajectory_df.shift(-1)
    trajectory_df, gdf_next = get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
    
    bearing_df = calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
    trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

    trajectory_df['dist'] = get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
    # Calculate the time difference between consecutive points
    time_differences = gdf_next['timestamp'] - trajectory_df['timestamp']
    
    # Calculate speed for points with subsequent points available
    speeds_mps = trajectory_df['dist'] / time_differences
    speeds_mps.fillna(0, inplace=True)
    
    trajectory_df['speed_mps'] = speeds_mps
    trajectory_df['speed_knots'] = trajectory_df['speed_mps'] * 1.943844
    
    return trajectory_df

def write_trajectories_for_area():
        # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        aalborg_havn_bbox:Polygon = box(minx=9.915161, miny=57.047827, maxx=10.061760, maxy=57.098774)
        
        # Assuming all necessary imports are already done
        #sparcify_trajectories_with_action_for_folder(folder_path=INPUT_AREA_TEST_FOLDER + '/realistic_strict', action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=aalborg_havn_bbox)
        #sparcify_trajectories_with_action_for_folder(folder_path=INPUT_AREA_TEST_FOLDER + '/realistic', action=sparcify_realisticly_trajectories, threshold=0.0, boundary_box=aalborg_havn_bbox)
        #sparcify_trajectories_with_action_for_folder(folder_path=INPUT_AREA_TEST_FOLDER + '/large_gap_0_5', action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=aalborg_havn_bbox)
        sparcify_trajectories_with_action_for_folder(folder_path=INPUT_AREA_TEST_FOLDER + '/random_0_5', action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=aalborg_havn_bbox)

def write_trajectories_for_all():
    # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        # Assuming all necessary imports are already done
        sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/realistic_strict', action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/realistic', action=sparcify_realisticly_trajectories, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/large_gap_0_5', action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/random_0_5', action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=None)

def is_directory_empty(directory):
    # Check if the directory is empty
    return not any(True for _ in os.listdir(directory))

def count_files(directory):
    # Count the number of files in the directory
    num_files = sum(1 for _ in os.listdir(directory) if os.path.isfile(os.path.join(directory, _)))
    return num_files

def check_empty_directories(root_dir):
    # Walk through the directory tree
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current directory is empty
        if is_directory_empty(dirpath):
            print("Empty directory:", dirpath)
        else:
            total_files += count_files(dirpath)

    print(total_files)

if __name__ == '__main__':
    freeze_support()

    # Assuming all necessary imports are already done
    sparcify_trajectories_with_action_for_folder(strfolder_path=INPUT_ALL_TEST_FOLDER + '/realistic_strict', action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=None)
    sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/realistic', action=sparcify_realisticly_trajectories, threshold=0.0, boundary_box=None)
    sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/large_gap_0_5', action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=None)
    sparcify_trajectories_with_action_for_folder(folder_path=INPUT_ALL_TEST_FOLDER + '/random_0_5', action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=None)