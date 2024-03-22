import os
import sys
import random
import time as t
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from shapely import wkb
from typing import Callable
from logs.logging import setup_logger
from shapely.geometry import box, Polygon
from multiprocessing import freeze_support  # Import freeze_support
from split_tractories import split_trajectories_from_df
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import utils

class SparsifyResult():
    reduced_points:int
    number_of_points:int
    trajectory_was_used:bool
    
    def __init__(self, reduced_points:int, number_of_points:int, trajectory_was_used:bool):
        self.reduced_points = reduced_points
        self.number_of_points = number_of_points
        self.trajectory_was_used = trajectory_was_used

AIS_CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'ais_csv')
ORIGINAL_FOLDER = os.path.join(os.path.dirname(__file__), 'original')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
INPUT_ALL_FOLDER = os.path.join(INPUT_FOLDER, 'all')
INPUT_ALL_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_ALL_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
INPUT_AREA_FOLDER = os.path.join(INPUT_FOLDER, 'area')
INPUT_AREA_VALIDATION_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'validation')
INPUT_AREA_TEST_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'test')
HARBORS_FILE = os.path.join(os.path.dirname(__file__), 'harbors.csv')
CSV_EXTRACT_FILE_LOG = 'ais_extraction_log.txt'


logging = setup_logger(CSV_EXTRACT_FILE_LOG)

def extract_trajectories_from_csv_files():
    #create_csv_file_for_mmsis(file_path=os.path.join(AIS_CSV_FOLDER,'aisdk-2024-02-11.csv')) #specify in method which mmsi
    #file_names = os.listdir(TEST_DATA_FOLDER)
    file_names = os.listdir(AIS_CSV_FOLDER)
    completed:int = 0
    
    logging.info(f'Began extracting trajectories from {len(file_names)} csv files')
    
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        #file_path = os.path.join(TEST_DATA_FOLDER, file_name)
        file_path = os.path.join(AIS_CSV_FOLDER, file_name)

        logging.info(f'Currently extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')        
        df:gpd.GeoDataFrame = cleanse_csv_file_and_convert_to_df(file_path)
        completed +=1
        
        logging.info(f'Finished extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')        
        logging.info(f'Began crating trajectories for file: {file_name}')

        if (not df.empty):
            create_trajectories_files(gdf=df)
        else:
            logging.warning(f'No data was extracted from {file_name}')
                
    logging.info('Finished creating trajecatories. Terminating')

def cleanse_csv_file_and_convert_to_df(file_path: str):
    """
    Takes a .csv file and cleanses it according to the set predicates.
    :param file_name: File name to cleanse. Example: 'aisdk-2022-01-01.csv'
    :return: A cleansed geodataframe, sorted by timestamp (ascending)
    """

    types = {
        '# Timestamp': str,
        'Type of mobile': str,
        'MMSI': 'Int32',
        'Navigational status': str,
        'Heading': 'Int16',
        'IMO': 'Int32',
        'Callsign': str,
        'Name': str,
        'Ship type': str,
        'Cargo type': str,
        'Width': 'Int32',
        'Length': 'Int32',
        'Type of position fixing device': str,
        'Destination': str,
        'ETA': str,
        'Data source type': str
    }
    
    df = pd.read_csv(file_path, na_values=['Unknown','Undefined'], dtype=types)#, nrows=1000000)    

    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type', 'Destination', 'Type of position fixing device',
                  'Callsign'],axis=1, errors='ignore')
           
    # Remove all the rows which does not satisfy our conditions
    df = df[
            (df["Type of mobile"] != "Class B") &
            (df["MMSI"].notna()) &
            (df["MMSI"].notnull()) &
            (df['# Timestamp'].notnull()) &
            (df['Latitude'] >=53.5) & (df['Latitude'] <=58.5) &
            (df['Longitude'] >= 3.2) & (df['Longitude'] <=16.5) &
            (df['SOG'] <=102)
    ].reset_index()
    
    subset_columns = ['MMSI', 'Latitude', 'Longitude', '# Timestamp']  # Adjust these based on your actual columns
    df = df.drop_duplicates(subset=subset_columns, keep='first')

    # We round the lat and longs as we do not need 15 decimals of precision
    # This will save some computation time later.
    # We also round rot, sog and cog, as we do not need a lot of decimal precision here
    df['Latitude'] = np.round(df['Latitude'], decimals=6)
    df['Longitude'] = np.round(df['Longitude'], decimals=6)
    df['ROT'] = np.round(df['ROT'], decimals=2)
    df['SOG'] = np.round(df['SOG'], decimals=2)
    df['COG'] = np.round(df['COG'], decimals=2)
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format="%d/%m/%Y %H:%M:%S").sort_values()
    df['# Timestamp'].astype('int64')//1e9
    df['# Timestamp'] = (df['# Timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()    

    # Rename the columns
    df = df.rename(columns={
            '# Timestamp':'timestamp',
            'Type of mobile':'type_of_mobile',
            'Navigational status':'navigational_status',
            'Ship type':'ship_type',
            'Type of position fixing device':'type_of_position_fixing_device',
        })

    # lower case names in the columns
    df.columns = map(str.lower, df.columns)
    
    if (df.empty):
        return df
    # Grouping by the columns 'imo', 'name', 'length', 'width', and 'ship_type'
    # and filling missing values within each group with the first non-null value
    df[['imo', 'name', 'length', 'width', 'ship_type']] = df.groupby('mmsi')[['imo', 'name', 'length', 'width', 'ship_type']].transform(lambda x: x.ffill())

    # Filling any remaining missing values with the last non-null value
    df[['imo', 'name', 'length', 'width', 'ship_type']] = df.groupby('mmsi')[['imo', 'name', 'length', 'width', 'ship_type']].transform(lambda x: x.bfill())
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    df = df.drop(columns=['index'], errors='ignore')
    df = df.to_crs(epsg="3857") # to calculate in meters
    
    return df
       
def create_trajectories_files(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return gdf
    
    harbors_df = extract_harbors_df()
    
    trajectories_df = gdf.sort_values('timestamp').groupby('mmsi')
    
    logging.info(f'Began creating trajectories for {trajectories_df.ngroups} mmsi')
    
    for mmsi, trajectory_df in trajectories_df:                
        sub_trajectories_df = split_trajectories_from_df(harbors_df=harbors_df, trajectory_df=trajectory_df) 
        
        if not sub_trajectories_df.empty:
            write_trajectories_to_original_folder(sub_trajectories_df)

def extract_harbors_df() -> gpd.GeoDataFrame:    
    df = pd.read_csv(HARBORS_FILE, na_values=['Unknown','Undefined']) 
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(bytes.fromhex(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.drop(columns=['index'], errors='ignore')
    gdf = gdf.to_crs(epsg="3857") # to calculate in meters
    
    return gdf

def write_trajectories_to_original_folder(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
            return
        
    for _, sub_trajectories in gdf.groupby('sub_trajectory_id'):
        if len(sub_trajectories) < 2:
            continue
        
        datetime_object = dt.datetime.fromtimestamp(sub_trajectories.iloc[0].timestamp, tz=dt.timezone.utc)
        str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-') 
        folder_name = str(sub_trajectories.iloc[0].mmsi)
        file_name = f'{folder_name}_{str_datetime}.txt'
        
        folder_path = os.path.join(ORIGINAL_FOLDER, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
       
        file_path = os.path.join(folder_path, file_name)
       
        write_trajectories(file_path, sub_trajectories)     
        
def write_trajectories(file_path:str, sub_trajectory: gpd.GeoDataFrame):
    sub_trajectory[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=True, mode='w')

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
                vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
                mmsi = root.split('/')[-1]
                new_folder_path = f'{ORIGINAL_FOLDER}/{vessel_folder}/{mmsi}'
                os.makedirs(new_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
            
                # only sog updated
                if not some_null_draught and not len(filtered_df) == len(trajectory_df):
                    datetime_object = dt.datetime.fromtimestamp(filtered_df.iloc[0].timestamp, tz=dt.timezone.utc)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-')            
                    file_name = f'{mmsi}_{str_datetime}.txt'
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                
                # nothing to update
                elif not some_null_draught and len(filtered_df) == len(trajectory_df):
                    file_name = file_name.replace('/', '-').replace(' ', '_').replace(':', '-') 
                    
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                
                # only druaght updated
                elif some_null_draught and len(filtered_df) == len(trajectory_df):
                    max_draught = filtered_df['draught'].max()
                    filtered_df['draught'] = filtered_df['draught'].fillna(max_draught)
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                    
                # both updated
                else:
                    max_draught = filtered_df['draught'].max()
                    filtered_df['draught'] = filtered_df['draught'].fillna(max_draught) # update draught
                    
                    # fix name, and remove old file
                    datetime_object = dt.datetime.fromtimestamp(filtered_df.iloc[0].timestamp, tz=dt.timezone.utc)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-')             
                    file_name = f'{mmsi}_{str_datetime}.txt'
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

                
                
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

        sparse_trajectory[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

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

            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            
            # Loop over points starting from the second one
            last_kept_index = 0
            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                speed_curr = speed_knots[i]
                keep_condition = (speed_last_kept >= 3 and speed_curr >= 3 and time_diff > 10) or \
                                (speed_last_kept >= 3  and (speed_curr < 3 or time_diff > 10)) or \
                                (speed_last_kept < 3 and speed_curr < 3 and time_diff > 180) or \
                                (speed_last_kept < 3 and (speed_curr >= 3 or time_diff > 180))

                # If the condition is false, mark the current point to be kept
                if keep_condition:
                    keep[i] = True
                    last_kept_index = i
            
            keep[0] = keep[-1] = True # keep first and last

            sparse_trajectory_df = trajectory_filtered_df[keep]
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')                
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].to_csv(new_file_path, sep=',', index=True, header=True, mode='w')     
    
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

            # Pre-allocate a boolean array to mark points to keep
            keep = np.full(shape=len(trajectory_filtered_df), fill_value=False, dtype=bool)
            
            # Loop over points starting from the second one
            last_kept_index = 0
            for i in range(1, len(trajectory_filtered_df)):
                time_diff = timestamps[i] - timestamps[last_kept_index]
                speed_last_kept = speed_knots[last_kept_index]
                speed_curr = speed_knots[i]
                keep_condition = (speed_last_kept >= 3 and speed_curr >= 3 and time_diff > 10) or \
                                (speed_last_kept >= 3 and time_diff > 10) or \
                                (speed_last_kept < 3 and speed_curr < 3 and time_diff > 180) or \
                                (speed_last_kept < 3 and time_diff > 180)

                # If the condition is false, mark the current point to be kept
                if keep_condition:
                    keep[i] = True
                    last_kept_index = i
            
            keep[0] = keep[-1] = True # keep first and last

            sparse_trajectory_df = trajectory_filtered_df[keep]
            
            # Drop the rows identified by the indices in the list            
            sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')        
            reduced_points = len(trajectory_filtered_df) - len(sparse_trajectory_df)
        else:
            if number_of_points == 2:
                trajectory_filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].to_csv(new_file_path, sep=',', index=True, header=True, mode='w')     
                
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
        sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'speed_mps', 'speed_knots']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

        reduced_points = len(trajectory_filtered) - len(sparse_trajectory_df)
        return SparsifyResult(reduced_points=reduced_points, number_of_points=number_of_points, trajectory_was_used=True)  
    except Exception as e:
        logging.error(f'Error occurred with: {repr(e)}')

def sparcify_trajectories_with_action_for_folder(
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
    trajectory_df, gdf_next = utils.get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
    
    bearing_df = utils.calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
    trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

    trajectory_df['dist'] = utils.get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
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

filter_original_trajectories(0.0)