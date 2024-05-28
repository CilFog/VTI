import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
from typing import List
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Polygon
from data.logs.logging import setup_logger
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CELL_TXT = os.path.join(DATA_FOLDER, 'cells.txt')
INPUT_GRAPH_FOLDER = os.path.join(DATA_FOLDER, 'input_graph')
INPUT_GRAPH_AREA_FOLDER = os.path.join(DATA_FOLDER, 'input_graph_area')
INPUT_GRAPH_CELLS_FOLDER = os.path.join(DATA_FOLDER, 'input_graph_cells')
INPUT_IMPUTATION_FOLDER = os.path.join(DATA_FOLDER, 'input_imputation')
INPUT_TEST_DATA_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'test')
INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER= os.path.join(INPUT_TEST_DATA_FOLDER, 'original_exam')
INPUT_TEST_SPARSED_FOLDER = os.path.join(INPUT_TEST_DATA_FOLDER, 'sparsed_exam')
INPUT_TEST_SPARSED_ALL_FOLDER = os.path.join(INPUT_TEST_SPARSED_FOLDER, 'all')

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
        return None

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

def sparcify_trajectories_with_meters_gaps_by_treshold(filepath:str, folderpath: str, threshold:float = 0.0):
    try:
        os_split = '/' if '/' in filepath else '\\'
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
        distances[-1] = 0   

        total_dist = distances.sum()
        

        if total_dist < threshold:
            return

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

    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')
        quit()

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
    
    return cell_txt_files_original_dict


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

vessels:dict = {
    'Fishing': [0, []],
    'Tanker': [0, []],
    'Tug': [0, []],
    'Cargo': [0, []],
    'Passenger': [0, []],
    'Dredging': [0, []],
    'Law_enforcement': [0, []],
    'Anti-pollution': [0, []],
    'Pilot': [0, []],
    'Pleasure': [0, []],
    'Towing': [0, []],
    'Port_tender': [0, []],
    'Diving': [0, []],
    'Towing_long_wide': [0, []],
}

cell_txt_files = find_cell_txt_files(cell_ids)
files = set(cell_txt_files.keys())

cells_df = pd.read_csv(CELL_TXT)
cells_df = cells_df[cells_df['cell_id'].isin(cell_ids)]
cells_df['geometry'] = cells_df.apply(lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']), axis=1)
cells_gdf = gpd.GeoDataFrame(cells_df, geometry='geometry', crs="EPSG:4326")

files = random.sample(files, len(files))

print('Finding files in cells')
for filepath in files:
    try:
        vessel = filepath.split('/')[-3]
        if vessel in vessels.keys():
            if vessels[vessel][0] == 5:
                continue

        trajectory_cell_df = get_trajectory_df_from_txt(filepath)
        
        if trajectory_cell_df is None or trajectory_cell_df.empty or not check_if_trajectory_is_dense(trajectory_cell_df):
            continue

        total_dist = trajectory_cell_df['dist'].sum()

        if (total_dist < 4000):
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
            continue

        largest_group_id = valid_groups.idxmax()

        # Filter trajectory points based on the largest group within the boundary box
        trajectory_cell_filtered_df = trajectory_cell_df[(trajectory_cell_df['group'] == largest_group_id) & trajectory_cell_df['within_cell']]

        # Filter trajectory points based on the largest group within the boundary box
        trajectory_cell_filtered_df = trajectory_cell_df[(trajectory_cell_df['group'] == largest_group_id) & trajectory_cell_df['within_cell']]
        total_dist = trajectory_cell_filtered_df['dist'][:-1]
        total_dist[-1] = 0
        dist_sum =  total_dist.sum() 
    
        if (not check_if_trajectory_is_dense(trajectory_cell_filtered_df) or dist_sum < 4000):
            continue

        if vessel in vessels.keys():
            vessels[vessel][0] += 1
            vessels[vessel][1].append(filepath)
        else:
            vessels[vessel] = (1, [filepath])
        
        # Collect the output strings in a list
        output_lines = []
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')

print('Found valid files. Began gathering original files')
for vessel in vessels.keys():
    try:
        if vessels[vessel][0] < 5:
            continue
        
        i = 0

        while i < 5:
            for (filepath) in vessels[vessel][1]:
                if i == 5:
                    break

                trajectory_cell_df = get_trajectory_df_from_txt(filepath)
                
                if (trajectory_cell_df is None or trajectory_cell_df.empty):
                    continue

                total_dist = trajectory_cell_df['dist'].sum()

                if (total_dist < 4000):
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
                                
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_cell_filtered_df = trajectory_cell_df[(trajectory_cell_df['group'] == largest_group_id) & trajectory_cell_df['within_cell']]
                total_dist = trajectory_cell_filtered_df['dist'][:-1].sum()

                if (not check_if_trajectory_is_dense(trajectory_cell_filtered_df) or total_dist < 4000):
                    continue
                
                path_split = '/'
                vessel_mmsi_folder = f'{filepath.split(path_split)[-3]}/{filepath.split(path_split)[-2]}'
                output_folder = INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER
                output_folder = os.path.join(output_folder, vessel_mmsi_folder)

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, os.path.basename(filepath))
                trajectory_cell_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 
                i += 1
    
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')


print('Finished gathering original files. Began creating test files')
test_files = Path(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER).rglob('*.txt')
test_files = [str(file) for file in test_files]

num_files = len(test_files)

for i,file in enumerate(test_files):
    try:
        sparcify_trajectories_with_meters_gaps_by_treshold(file, INPUT_TEST_SPARSED_ALL_FOLDER, 4000)
        print(f'\rFinished {i}/{num_files} files')
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')