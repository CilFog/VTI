import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List
from shapely.geometry import box, Polygon
from data.logs.logging import setup_logger
from data.stats.statistics import Sparse_Statistics
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters
from .classes import brunsbuettel_to_kiel_polygon, aalborg_harbor_to_kattegat_bbox, doggersbank_to_lemvig_bbox, skagens_harbor_bbox, cells_bbox
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CELL_TXT = os.path.join(DATA_FOLDER, 'cells.txt')
INPUT_GRAPH_FOLDER = os.path.join(DATA_FOLDER, 'input_graph')
INPUT_GRAPH_AREA_FOLDER = os.path.join(DATA_FOLDER, 'input_graph_area')
INPUT_GRAPH_CELLS_FOLDER = os.path.join(DATA_FOLDER, 'input_graph_cells')
INPUT_IMPUTATION_FOLDER = os.path.join(DATA_FOLDER, 'input_imputation')
INPUT_TEST_DATA_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'test')
INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER= os.path.join(INPUT_TEST_DATA_FOLDER, 'original')
INPUT_TEST_SPARSED_FOLDER = os.path.join(INPUT_TEST_DATA_FOLDER, 'sparsed')
INPUT_TEST_SPARSED_ALL_FOLDER = os.path.join(INPUT_TEST_SPARSED_FOLDER, 'all')
INPUT_TEST_SPARSED_AREA_FOLDER = os.path.join(INPUT_TEST_SPARSED_FOLDER, 'area')
INPUT_VALIDATION_DATA_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'validation')
INPUT_VALIDATION_DATA_ORIGINAL_FOLDER = os.path.join(INPUT_VALIDATION_DATA_FOLDER, 'original')
INPUT_VALIDATION_SPARSED_FOLDER = os.path.join(INPUT_VALIDATION_DATA_FOLDER, 'sparsed')
INPUT_VALIDATION_SPARSED_ALL_FOLDER = os.path.join(INPUT_VALIDATION_SPARSED_FOLDER, 'all')
INPUT_VALIDATION_SPARSED_AREA_FOLDER = os.path.join(INPUT_VALIDATION_SPARSED_FOLDER, 'area')
INPUT_VALIDATION_SPARSED2_FOLDER = os.path.join(INPUT_VALIDATION_DATA_FOLDER, 'sparsed2')
INPUT_VALIDATION_SPARSED2_ALL_FOLDER = os.path.join(INPUT_VALIDATION_SPARSED2_FOLDER, 'all')
INPUT_VALIDATION_SPARSED2_AREA_FOLDER = os.path.join(INPUT_VALIDATION_SPARSED2_FOLDER, 'area')

STATS_FOLDER = os.path.join(DATA_FOLDER, 'stats')
STATS_INPUT_IMPUTATION = os.path.join(STATS_FOLDER, 'input_imputation')
STATS_TEST = os.path.join(STATS_INPUT_IMPUTATION, 'test')
STATS_TEST_ALL = os.path.join(STATS_TEST, 'all')
STATES_TEST_AREA = os.path.join(STATS_TEST, 'area')
STATS_VALIDATION = os.path.join(STATS_INPUT_IMPUTATION, 'validation')
STATS_VALIDATION_ALL = os.path.join(STATS_VALIDATION, 'all')
STATS_VALIDATION_AREA = os.path.join(STATS_VALIDATION, 'area')
STATS_TEST_THRESHOLD = os.path.join(STATS_TEST, 'stats_threshold.csv')
STATS_TEST_REALISTIC = os.path.join(STATS_TEST, 'stats_realistic.csv')
STATS_VALIDATION_THRESHOLD = os.path.join(STATS_VALIDATION, 'stats_threshold.csv')
STATS_VALIDATION_REALISTIC = os.path.join(STATS_VALIDATION, 'stats_realistic.csv')
STATS_VALIDATION2_THRESHOLD = os.path.join(STATS_VALIDATION, 'stats_threshold2.csv')
STATS_VALIDATION2_REALISTIC = os.path.join(STATS_VALIDATION, 'stats_realistic2.csv')
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

def sparcify_trajectories_with_meters_gaps_by_treshold(filepath:str, folderpath: str, stats, output_json:str, threshold:float = 0.0, boundary_box:Polygon = None):
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_points:int = 0
        number_of_vessel_samples:int = 0
                                
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        gaps_folder =  os.path.join(folderpath, 'many_gap')
        gaps_folder = os.path.join(gaps_folder, f'{threshold}'.replace('.', '_'))
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(gaps_folder, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        
        trajectory_df['group'] = 0
        
        distances = trajectory_df['dist'].to_numpy()

        total_dist = distances.sum()

        if total_dist < threshold:
            return

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
                distances = trajectory_filtered_df['dist'].to_numpy()
                distances[-1] = 0
                total_dist = distances.sum()

                if total_dist < threshold:
                    return
                
                # Update total number of points
                number_of_vessel_samples = len(trajectory_filtered_df)
            else:
                # No valid groups within the boundary box
                return
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            number_of_vessel_samples = len(trajectory_filtered_df)
        
        if number_of_vessel_samples < 2:
            return        

        keep = np.full(shape=number_of_vessel_samples, fill_value=False, dtype=bool)
   
        curr_distance = 0
        prev_point = 0
        for i in range(1, number_of_vessel_samples):
            curr_distance += distances[prev_point]
            prev_point = i
            if (curr_distance > threshold):
                keep[i] = True
                curr_distance = 0

        keep[0] = keep[-1] = True # keep first and last

        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        sparse_trajectory_df = trajectory_filtered_df[keep]
        sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)    
        
        stats.output_folder = gaps_folder
        stats.threshold = threshold
        stats.vessel_samples = number_of_vessel_samples
        stats.reduced = reduced_points
        stats.total_distance = total_dist
        stats.vessel_type = vessel_folder
        stats.add_to_file(output_json)

    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
        quit()

def sparcify_trajectories_realisticly(filepath:str, folderpath: str, stats, output_json:str, boundary_box:Polygon = None):
    try:    
        os_split = '/' if '/' in filepath else '\\'
        reduced_vessel_samples:int = 0
        number_of_vessel_samples:int = 0
        
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, 'realistic')
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        trajectory_df['group'] = 0
        
        total_dist = trajectory_df['dist'].sum()

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
                number_of_vessel_samples = len(trajectory_filtered_df)
            else:
                # No valid groups within the boundary box
                return
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            total_dist = trajectory_filtered_df['dist'].sum()
            number_of_vessel_samples = len(trajectory_filtered_df)
            if number_of_vessel_samples < 2:
                return
        
        if number_of_vessel_samples < 2:
            return       
        
        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        if number_of_vessel_samples > 2:
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
            
            
            reduced_vessel_samples = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_filepath, sep=',', index=True, header=True, mode='w')     
        
        stats.output_folder = folderpath
        stats.threshold = 0
        stats.vessel_samples = number_of_vessel_samples
        stats.reduced = reduced_vessel_samples
        stats.total_distance = total_dist
        stats.vessel_type = vessel_folder
        stats.add_to_file(output_json)

    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')    
        quit()

def sparcify_realisticly_strict_trajectories(filepath:str, folderpath: str, stats, output_json:str, boundary_box:Polygon = None):
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_vessel_samples:int = 0
        number_of_vessel_samples:int = 0
  
        trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        folderpath = os.path.join(folderpath, 'realistic_strict')
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(folderpath, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)
        
        trajectory_df['group'] = 0
        total_dist = trajectory_df['dist'].sum()
        
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
                number_of_vessel_samples = len(trajectory_filtered_df)

                total_dist = trajectory_filtered_df['dist'].sum()
            else:
                # No valid groups within the boundary box
                return
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            number_of_vessel_samples = len(trajectory_filtered_df)
        
        if number_of_vessel_samples < 2:
            return
        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        if number_of_vessel_samples > 2:
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
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')        
            reduced_vessel_samples = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            if number_of_vessel_samples == 2:
                trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].to_csv(new_filepath, sep=',', index=True, header=True, mode='w')     
                
        stats.output_folder = folderpath
        stats.threshold = 0
        stats.vessel_samples = number_of_vessel_samples
        stats.reduced = reduced_vessel_samples
        stats.total_distance = total_dist
        stats.vessel_type = vessel_folder
        stats.add_to_file(output_json)
    except Exception as e:
        print(e)
        quit()

def sparcify_large_meter_gap_by_threshold(filepath:str, folderpath: str, stats, output_json:str, threshold:float = 0.0, boundary_box:Polygon = None):
    try:
        os_split = '/' if '/' in filepath else '\\'
        reduced_vessel_samples = 0
        number_of_vessel_samples = 0
          
        trajectory_df = get_trajectory_df_from_txt(file_path=filepath)
        filename = os.path.basename(filepath)
        gap_folder =  os.path.join(folderpath, 'single_gap')
        gap_folder = os.path.join(gap_folder, f'{threshold}'.replace('.', '_'))
        vessel_folder = filepath.split(os_split)[-3]
        vessel_folder_path = os.path.join(gap_folder, vessel_folder)
        new_filepath = os.path.join(vessel_folder_path, filename)

        distances = trajectory_df['dist'].to_numpy()

        total_dist = distances.sum()

        if total_dist < threshold:
            return

        trajectory_df['group'] = 0
        
        if boundary_box:
            gdf_bbox = gpd.GeoDataFrame([1], geometry=[boundary_box], crs="EPSG:4326").geometry.iloc[0] 
            trajectory_df['within_boundary_box'] = trajectory_df.within(gdf_bbox)            
            change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(-1)
            trajectory_df['group'] = change_detected.cumsum()
            
            # Find the largest group within the boundary box
            group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
            valid_groups = group_sizes[group_sizes >= 2]

            if not valid_groups.empty:
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered_df = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]
                distances = trajectory_filtered_df['dist'].to_numpy()
                distances[-1] = 0
                total_dist = distances.sum()

                if total_dist < threshold:
                    return
                
                # Update total number of points
                number_of_vessel_samples = len(trajectory_filtered_df)

            else:
                # No valid groups within the boundary box
                return
        else:
            # If no boundary box is provided, use the original dataframe
            trajectory_filtered_df = trajectory_df
            number_of_vessel_samples = len(trajectory_filtered_df)
            
        if number_of_vessel_samples < 2:
            return
        keep = np.full(shape=number_of_vessel_samples, fill_value=False, dtype=bool)

        curr_distance = 0
        prev_point = 0
        for i in range(1, number_of_vessel_samples):
            curr_distance += distances[prev_point]
            prev_point = i
            if (curr_distance > threshold):
                keep[i] = True
                keep[i+1:] = [True] * (number_of_vessel_samples - i - 1)  # Set all subsequent entries to True
                break

        keep[0] = keep[-1] = True # keep first and last

        os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        sparse_trajectory_df = trajectory_filtered_df[keep]
        sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_filepath, sep=',', index=True, header=True, mode='w')

        reduced_vessel_samples = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        
        stats.output_folder = gap_folder
        stats.threshold = threshold
        stats.vessel_samples = number_of_vessel_samples
        stats.reduced = reduced_vessel_samples
        stats.total_distance = total_dist
        stats.vessel_type = vessel_folder
        stats.add_to_file(output_json)

    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
        quit()

def check_if_trajectory_is_dense(trajectory_df: pd.DataFrame) -> bool:
    try:    
        # Convert timestamps to numpy array for efficient calculations
        timestamps = trajectory_df['timestamp'].to_numpy()

        # Calculate the time differences between consecutive points
        time_diffs = np.diff(timestamps)

        # Check if all differences are less than or equal to the 30 seconds
        return np.all(time_diffs <= 30)
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')    
        quit()

def write_trajectories_for_area(file:str, outputfolder: str, output_json:str):
    aalborg_harbor_to_kattegat_path = os.path.join(outputfolder, 'aalborg_harbor_to_kattegat')
    skagen_harbor_path = os.path.join(outputfolder, 'skagen_harbor')
    cells_path = os.path.join(outputfolder, 'cells')
    
    outputfolders = [
        (aalborg_harbor_to_kattegat_path, aalborg_harbor_to_kattegat_bbox),
        (skagen_harbor_path, skagens_harbor_bbox),
        (cells_path, cells_bbox)]
    
    if ('test' not in outputfolder):
        brunsbuettel_to_kiel_path = os.path.join(outputfolder, 'brunsbuettel_to_kiel')
        doggersbank_to_lemvig_path = os.path.join(outputfolder, 'doggersbank_to_lemvig')
        
        outputfolders.append((brunsbuettel_to_kiel_path, brunsbuettel_to_kiel_polygon))
        outputfolders.append((doggersbank_to_lemvig_path, doggersbank_to_lemvig_bbox))
    
    stats = Sparse_Statistics()
    for (outputfolder, boundary_box) in outputfolders:
        sparcify_trajectories_realisticly(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, boundary_box=boundary_box)

    logging.info(f'Finished all area in output folder {outputfolder}')   

def write_trajectories_for_all(file: str, outputfolder:str, output_json:str):
    stats = Sparse_Statistics()
    sparcify_trajectories_realisticly(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, boundary_box=None)

def write_trajectories_for_area_with_threshold(file:str, outputfolder: str, threshold:float, output_json:str):
    aalborg_harbor_to_kattegat_path = os.path.join(outputfolder, 'aalborg_harbor_to_kattegat')
    skagen_harbor_path = os.path.join(outputfolder, 'skagen_harbor')
    cells_path = os.path.join(outputfolder, 'cells')

    outputfolders = [
        (aalborg_harbor_to_kattegat_path, aalborg_harbor_to_kattegat_bbox),
        (skagen_harbor_path, skagens_harbor_bbox),
        (cells_path, cells_bbox)]
    
    if ('test' not in outputfolder):
        brunsbuettel_to_kiel_path = os.path.join(outputfolder, 'brunsbuettel_to_kiel')
        doggersbank_to_lemvig_path = os.path.join(outputfolder, 'doggersbank_to_lemvig')

        outputfolders.append((brunsbuettel_to_kiel_path, brunsbuettel_to_kiel_polygon))
        outputfolders.append((doggersbank_to_lemvig_path, doggersbank_to_lemvig_bbox))
    
    stats = Sparse_Statistics()
    for (outputfolder, boundary_box) in outputfolders:
        sparcify_large_meter_gap_by_threshold(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, threshold=threshold, boundary_box=boundary_box)
        sparcify_trajectories_with_meters_gaps_by_treshold(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, threshold=threshold, boundary_box=boundary_box)

def write_trajectories_for_all_with_threshold(file: str, outputfolder:str, threshold:float, output_json:str):
    stats = Sparse_Statistics()

    sparcify_large_meter_gap_by_threshold(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, threshold=threshold, boundary_box=None)
    sparcify_trajectories_with_meters_gaps_by_treshold(filepath=file, folderpath=outputfolder, stats=stats, output_json=output_json, threshold=threshold, boundary_box=None)

    logging.info(f'Finished all for threshold {threshold} in output folder {outputfolder}')   

def find_cell_txt_files(directories: List[str]):
    cell_txt_files:list[str] = []

    for directory in directories:
        cell_directory = os.path.join(INPUT_GRAPH_CELLS_FOLDER, directory)
        path_obj = Path(cell_directory)
        cell_txt_files.extend(path_obj.rglob('*.txt'))
    
    cell_txt_files_original_dict:dict = {}
    for path in cell_txt_files:
        path_str = str(path)
        for cell_id in directories:
            if cell_id in path_str:
                # Replace everything up to and including the cell ID
                updated_path = path_str.split(cell_id, 1)[-1]
                updated_path = updated_path.lstrip("/")  # Remove any leading slashes
                updated_path = os.path.join(INPUT_GRAPH_FOLDER, updated_path)
                
                if updated_path not in cell_txt_files_original_dict.keys():
                    cell_txt_files_original_dict[updated_path] = [path_str]
                else:
                    cell_txt_files_original_dict[updated_path].append(path_str)
    
    cell_txt_files_original = [(filepath, cellpaths) for filepath, cellpaths in cell_txt_files_original_dict.items() for path in cellpaths]
    return cell_txt_files_original

def move_random_files_to_test_and_validation(percentage=0.1):
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'
    directories_with_moved_files = set()
    
    print('Getting all input files')
    all_files = list(Path(INPUT_GRAPH_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    all_files = [str(path) for path in all_files] # Convert Path objects to strings

    print('Calculating number of files to move to validation')
    num_files_to_move_to_validation = int(len(all_files) * percentage)
    
    not_dense_files:list[str] = []
    files_moved:list[str] = []

    # for test
    print('Began working on test files')
    try:
        cell_ids = [
            '9_9', 
            '9_10', 
            '9_11', 
            '10_9', 
            '10_10', 
            '10_11', 
            '11_9', 
            '11_10', 
            '11_11'
        ]
        
        print('get cells txt files')
        # get files (vessel samples) within all cells
        cell_txt_files = find_cell_txt_files(cell_ids)
        
        # read cells in txt
        cells_df = pd.read_csv(CELL_TXT)

        print('preparing grid cells in df')
        # filter cells not in the cells array
        cells_df = cells_df[cells_df['cell_id'].isin(cell_ids)]
        cells_df['geometry'] = cells_df.apply(lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']), axis=1)
        cells_gdf = gpd.GeoDataFrame(cells_df, geometry='geometry', crs="EPSG:4326")
        
        print('calculating number of files to move to test')
        # calculate the number of files to move to test given the percentage
        num_files_to_move_to_test = int(len(cell_txt_files) * percentage)
        logging.info(f'Began moving test files {num_files_to_move_to_test}')
        
        # select random files
        random_files_to_move = random.sample(cell_txt_files, num_files_to_move_to_test)
        
        num_files_moved_to_test = 0

        while(num_files_moved_to_test < num_files_to_move_to_test):
            for (filepath, cell_paths) in random_files_to_move:
                # check we are not done
                if (num_files_moved_to_test >= num_files_to_move_to_test):
                    break
                
                # ensure we do not move the same file twice
                if (filepath in files_moved or filepath in not_dense_files):
                    continue

                trajectory_cell_df = get_trajectory_df_from_txt(filepath)

                if (trajectory_cell_df is None or trajectory_cell_df.empty):
                    not_dense_files.append(filepath)
                    continue

                # spatial join: find which points fall within any of the cells
                joined_gdf = gpd.sjoin(trajectory_cell_df, cells_gdf, how='left', predicate='within')

                # create the boolean column based on whether a cell was matched (i.e., `index_right` is not NaN)
                trajectory_cell_df['within_cell'] = ~joined_gdf['index_right'].isna()     
                change_detected = trajectory_cell_df['within_cell'] != trajectory_cell_df['within_cell'].shift(1)
                trajectory_cell_df['group'] = change_detected.cumsum()
            
                # find the largest group within the boundary box
                group_sizes = trajectory_cell_df[trajectory_cell_df['within_cell']].groupby('group').size()
                valid_groups = group_sizes[group_sizes >= 2]

                if valid_groups.empty:
                    not_dense_files.append(filepath)
                    continue
                    
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_cell_filtered_df = trajectory_cell_df[(trajectory_cell_df['group'] == largest_group_id) & trajectory_cell_df['within_cell']]

                if (not check_if_trajectory_is_dense(trajectory_cell_filtered_df)):
                    not_dense_files.append(filepath)
                    continue

                vessel_mmsi_folder = f'{filepath.split(os_path_split)[-3]}/{filepath.split(os_path_split)[-2]}'
                output_folder = INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER
                output_folder = os.path.join(output_folder, vessel_mmsi_folder)

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, os.path.basename(filepath))
                trajectory_cell_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 
                
                # remove file from input graph
                files_moved.append(filepath)
                num_files_moved_to_test += 1
                os.remove(filepath)
                directories_with_moved_files.add(os.path.dirname(filepath))

                for (cell_path) in cell_paths:
                    os.remove(cell_path)
                    directories_with_moved_files.add(os.path.dirname(cell_path))

                sys.stdout.write(f"\rMoved {num_files_moved_to_test}/{num_files_to_move_to_test} to test")
                sys.stdout.flush()

            if (num_files_moved_to_test < num_files_to_move_to_test):
                random_files_to_move = random.sample(cell_txt_files, num_files_to_move_to_test)

    except Exception as e:
        logging.error('Error was thrown with', repr(e))
    
    logging.info(f'\nFinished moving {num_files_moved_to_test} to test')

    num_files_moved_to_validation = 0

    # randomly select the files
    random_files_to_move = random.sample(all_files, num_files_to_move_to_validation)

    logging.info(f'Began moving files to validation {num_files_to_move_to_validation}')
    try:
        while (num_files_moved_to_validation < num_files_to_move_to_validation):
            for filepath in random_files_to_move:
                
                # check if we have moved the desired number of files
                if num_files_moved_to_validation >= num_files_to_move_to_validation: 
                    break

                # check if we have already moved the file or if it is not dense
                if filepath in files_moved or filepath in not_dense_files:
                    continue

                trajectory_df = get_trajectory_df_from_txt(filepath)

                if (not check_if_trajectory_is_dense(trajectory_df)):
                    not_dense_files.append(filepath)
                    continue

                # move the file to input imputation folder with vessel/mmsi folder structure
                vessel_mmsi_folder = f'{filepath.split(os_path_split)[-3]}/{filepath.split(os_path_split)[-2]}'

                # move the file to validation folder 
                end_dir = os.path.join(INPUT_VALIDATION_DATA_ORIGINAL_FOLDER, vessel_mmsi_folder)
                os.makedirs(end_dir, exist_ok=True)
                shutil.move(filepath, end_dir)
                directories_with_moved_files.add(os.path.dirname(filepath))
                num_files_moved_to_validation += 1
                sys.stdout.write(f"\rMoved {num_files_moved_to_validation}/{num_files_to_move_to_validation} to validation")
                sys.stdout.flush()
                files_moved.append(filepath)

            if (num_files_moved_to_validation < num_files_to_move_to_validation):
                random_files_to_move = random.sample(all_files, num_files_to_move_to_validation)

        logging.info(f'\nFinished moving {num_files_moved_to_validation} to validation')

    except Exception as e:
        logging.error(f'Error was thrown with {repr(e)}')

    logging.info('Began removing empty directories')
    empty_folders_removed = 0
    for dir_path in directories_with_moved_files:
        try:
            if not os.listdir(dir_path):  # Check if directory is empty
                os.rmdir(dir_path)  # Remove empty directory
                dir_dir_path = os.path.dirname(dir_path) # remove parent i empty
                if not os.listdir(dir_dir_path):
                    os.rmdir(dir_dir_path)

                empty_folders_removed += 1
        except Exception as e:
            logging.error(f'Error was thrown with {repr(e)} for files in {dir_path}')

    logging.info(f'Finished moving {num_files_moved_to_validation + num_files_moved_to_test} files\n Removed {empty_folders_removed} empty directories from input graph')

def find_area_input_files():
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'

    all_files = list(Path(INPUT_GRAPH_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    all_files = [str(path) for path in all_files] # Convert Path objects to strings

    brunsbuettel_to_kiel_gdf = gpd.GeoDataFrame([1], geometry=[brunsbuettel_to_kiel_polygon], crs="EPSG:4326").geometry.iloc[0]
    aalborg_harbor_to_kattegat_gdf = gpd.GeoDataFrame([1], geometry=[aalborg_harbor_to_kattegat_bbox], crs="EPSG:4326").geometry.iloc[0]
    doggersbank_to_lemvig_gdf = gpd.GeoDataFrame([1], geometry=[doggersbank_to_lemvig_bbox], crs="EPSG:4326").geometry.iloc[0] 
    skagen_gdf = gpd.GeoDataFrame([1], geometry=[skagens_harbor_bbox], crs="EPSG:4326").geometry.iloc[0]

    areas = [
        (brunsbuettel_to_kiel_gdf, 'brunsbuettel_to_kiel'), 
        (aalborg_harbor_to_kattegat_gdf, 'aalborg_harbor_to_kattegat'), 
        (doggersbank_to_lemvig_gdf, 'doggersbank_to_lemvig'),
        (skagen_gdf, 'skagen_harbor')]

    logging.info(f'Began finding area input files for {len(all_files)} files')
    for file_path in all_files:
        try:
            for (area, name) in areas:
                trajectory_df = get_trajectory_df_from_txt(file_path)
                trajectory_df['within_boundary_box'] = trajectory_df.within(area)            
                change_detected = trajectory_df['within_boundary_box'] != trajectory_df['within_boundary_box'].shift(1)
                trajectory_df['group'] = change_detected.cumsum()
                
                # Find the largest group within the boundary box
                group_sizes = trajectory_df[trajectory_df['within_boundary_box']].groupby('group').size()
                valid_groups = group_sizes[group_sizes >= 2]

                if valid_groups.empty:
                    continue
            
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered_df = trajectory_df[(trajectory_df['group'] == largest_group_id) & trajectory_df['within_boundary_box']]

                vessel_mmsi_folder = f'{file_path.split(os_path_split)[-3]}/{file_path.split(os_path_split)[-2]}'
                output_folder = INPUT_GRAPH_AREA_FOLDER
                output_folder = os.path.join(output_folder, name)
                output_folder = os.path.join(output_folder, vessel_mmsi_folder)

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, os.path.basename(file_path))
                trajectory_filtered_df.reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 
        
        except Exception as e:
            logging.error(f'Error was thrown with {repr(e)} for file {file_path}')       

    logging.info('Finished finding area input files')        

def find_cell_input_files():
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'

    cells_df = pd.read_csv(CELL_TXT)
    cells_df['geometry'] = cells_df.apply(lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']), axis=1)
    cells_gdf = gpd.GeoDataFrame(cells_df, geometry='geometry', crs="EPSG:4326")

    all_files = list(Path(INPUT_GRAPH_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    all_files = [str(path) for path in all_files] # Convert Path objects to strings

    num_files = len(all_files)

    logging.info(f'Began finding area input files for {len(all_files)} files')
    i = 1;
    for file_path in all_files:
        try:
            trajectory_df = get_trajectory_df_from_txt(file_path)
            
            # Perform a spatial join to identify points from trajectory_df that intersect harbors
            points_in_cells = gpd.sjoin(trajectory_df, cells_gdf, how="left", predicate="intersects", lsuffix='left', rsuffix='right')

            for (cell_id, group) in points_in_cells.groupby('cell_id'):
                if group.empty:
                    continue

                vessel_mmsi_folder = f'{file_path.split(os_path_split)[-3]}/{file_path.split(os_path_split)[-2]}'
                output_folder = INPUT_GRAPH_CELLS_FOLDER
                output_folder = os.path.join(output_folder, str(cell_id))
                output_folder = os.path.join(output_folder, vessel_mmsi_folder)

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, os.path.basename(file_path))
                group[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 
            
            sys.stdout.write(f"\rCell data created for {i}/{num_files} trajectories")
            sys.stdout.flush()
            i += 1
        
        except Exception as e:
            logging.error(f'Error was thrown with {repr(e)} for file {file_path}')       

    logging.info('Finished finding area input files')        

def make_test_and_validation_data():
    threshold_values = [500, 1000, 2000, 4000, 8000]
    # test_all_threshold = os.path.join(STATS_TEST_ALL, 'all_threshold.json')
    # test_area_threshold = os.path.join(STATES_TEST_AREA, 'area_threshold.json')
    # test_all_realistic = os.path.join(STATS_TEST_ALL, 'all_realistic.json')
    # test_area_realistic = os.path.join(STATES_TEST_AREA, 'area_realistic.json')
    validation_all_threshold = os.path.join(STATS_VALIDATION_ALL, 'all_threshold2.json')
    validation_area_threshold = os.path.join(STATS_VALIDATION_AREA, 'area_threshold2.json')
    validation_all_realistic = os.path.join(STATS_VALIDATION_ALL, 'all_realistic2.json')
    validation_area_realistic = os.path.join(STATS_VALIDATION_AREA, 'area_realistic2.json')

    # os.makedirs(STATS_TEST_ALL, exist_ok=True)
    # os.makedirs(STATES_TEST_AREA, exist_ok=True)
    os.makedirs(STATS_VALIDATION_ALL, exist_ok=True)
    os.makedirs(STATS_VALIDATION_AREA, exist_ok=True)
    stats = Sparse_Statistics()

    print('gettings vessels folders')
    vessel_folders = [folder for folder in Path(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER).iterdir() if folder.is_dir()]
    # Traverse 
    print('making data for validation')
    for folder in vessel_folders:
        if (folder.name.lower() == 'fishing'):
            continue
        vessel_files = list(folder.rglob('*.txt'))
        vessel_files = [str(path) for path in vessel_files]
        random_files = []
        try:
            random_files = random.sample(vessel_files, 50)
        except Exception:
            random_files = vessel_files
        
        for file in random_files:
            for threshold in threshold_values:
                write_trajectories_for_all_with_threshold(file, INPUT_VALIDATION_SPARSED2_ALL_FOLDER, threshold=threshold, output_json=validation_all_threshold)
                write_trajectories_for_area_with_threshold(file, INPUT_VALIDATION_SPARSED2_AREA_FOLDER, threshold=threshold, output_json=validation_area_threshold)

            write_trajectories_for_all(file, INPUT_VALIDATION_SPARSED2_ALL_FOLDER, output_json=validation_all_realistic)
            write_trajectories_for_area(file, INPUT_VALIDATION_SPARSED2_AREA_FOLDER, output_json=validation_area_realistic)

    print('making stats for validation')
    stats.make_statistics_with_threshold(validation_all_threshold)
    stats.make_statistics_with_threshold(validation_area_threshold)
    stats.make_statistics_no_threshold(validation_all_realistic)
    stats.make_statistics_no_threshold(validation_area_realistic)

make_test_and_validation_data()