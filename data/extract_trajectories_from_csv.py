import os
import sys
import inspect
import time as t
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from shapely import wkb
from split_tractories import split_trajectories_from_df
from logs.logging import setup_logger

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import utils

AIS_CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'ais_csv')
ORIGINAL_FOLDER = os.path.join(os.path.dirname(__file__), 'original')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
INPUT_ALL_FOLDER = os.path.join(INPUT_FOLDER, 'all')
INPUT_ALL_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_ALL_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
INPUT_AREA_FOLDER = os.path.join(INPUT_FOLDER, 'all')
INPUT_AREA_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_AREA_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
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
        
        datetime_object = dt.datetime.utcfromtimestamp(sub_trajectories.iloc[0].timestamp)
        str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_')
        folder_name = str(sub_trajectories.iloc[0].mmsi)
        file_name = f'{folder_name}_{str_datetime}.txt'
        
        folder_path = os.path.join(ORIGINAL_FOLDER, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
       
        file_path = os.path.join(folder_path, file_name)
       
        write_trajectories(file_path, sub_trajectories)     

def write_trajectories_for_area(file_name: str, sub_trajectories: gpd.GeoDataFrame):
    return

def write_trajectories_for_area_all(file_name: str, sub_trajectories: gpd.GeoDataFrame):
    return
        
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
        df = df.to_crs(epsg="3857") # to calculate in meters
        return df
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')

def filter_original_trajectories(sog_threshold: float):
    removed_ship_type = 0
    removed_sog = 0
    removed_draught = 0
    ship_types = ['fishing', 'tanker', 'tug', 'cargo', 'passenger', 'dredging', 'law enforcement', 'anti-pollution', 'pilot', 'pleasure', 'towing', 'port tender', 'diving', 'towing long/wide', ''] 
    moved = 0
    initial_num = 0
    deleted = 0
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
                    
                some_null_draught = filtered_df.draught.isnull().any() or filtered_df.draught.isna().any()
                
                file_name = file.split('/')[-1]
                vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
                mmsi = root.split('/')[-1]
                new_folder_path = f'{ORIGINAL_FOLDER}/{vessel_folder}/{mmsi}'
                os.makedirs(new_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

                # only sog updated
                if not some_null_draught and not len(filtered_df) == len(trajectory_df):
                    datetime_object = dt.datetime.utcfromtimestamp(filtered_df.iloc[0].timestamp)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_')            
                    file_name = f'{mmsi}_{str_datetime}.txt'
                    new_file_path = os.path.join(new_folder_path, file_name)
                    filtered_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')
                
                # nothing to update
                elif not some_null_draught and len(filtered_df) == len(trajectory_df):
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
                    datetime_object = dt.datetime.utcfromtimestamp(filtered_df.iloc[0].timestamp)
                    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_')            
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
    
    removed = removed_draught + removed_ship_type + removed_sog
    
    finished_time = t.perf_counter() - initial_time
    logging.info(f'\nRemoved_due_to_ship: {removed_ship_type}\nRemoved_due_to_sog: {removed_sog}\nRemoved_due_to_draught: {removed_draught}\nTotal removed: ({removed}/{initial_num})\nTotal moved to new location: ({moved}/{initial_num})\nElapsed time: {finished_time:0.4f} seconds"')   

def sparcify_original_trajectories_80_percent_for_folder(folder_path: str):
    initial_time = t.perf_counter()
    removed = 0
    removed_avg = 0
    num_files = 0
    total_number_of_points = 0
    logging.info('Began sparcifying original trajectories') 
     
    for root, dirs, files in os.walk(ORIGINAL_FOLDER):
        num_files += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
            
            file_name = file.split('/')[-1]
            vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
            vessel_folder_path = f'{folder_path}/{vessel_folder}'
            new_file_path = os.path.join(vessel_folder_path, file_name)
            
            os.makedirs(vessel_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

            # Mark ~80% rows for removal
            # Ensure first and last rows are not marked for removal
            rows_to_remove = np.random.choice([True, False], size=len(trajectory_df), p=[0.8, 0.2])
            
            # Ensure first and last isn't removed
            rows_to_remove[0] = rows_to_remove[-1] = False
            
            # Reindex rows_to_remove to match trajectory_df's index
            rows_to_remove = rows_to_remove.reindex_like(trajectory_df)
            
            # Sparsify
            sparsified_trajectory = trajectory_df[~rows_to_remove]
            sparsified_trajectory.reset_index(drop=True, inplace=True)
            
            sparsified_trajectory[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(new_file_path, sep=',', index=True, header=True, mode='w')

            total_number_of_points += len(trajectory_df)
            removed += len(trajectory_df) - len(sparsified_trajectory)
            
    finished_time = t.perf_counter() - initial_time
    removed_avg = removed/num_files
    logging.info(f'Removed on avg. pr trajectory: {removed_avg}. Removed in total: {removed}/{total_number_of_points}. Elapsed time: {finished_time:0.4f} seconds')   

def sparcify_realisticly_original_trajectories(folder_path: str):
    initial_time = t.perf_counter()
    removed = 0
    removed_avg = 0
    num_files = 0
    total_number_of_points = 0
    logging.info('Began sparcifying original trajectories') 
     
    for root, dirs, files in os.walk(ORIGINAL_FOLDER):
        num_files += len(files)
        
        for file in files:
            file_path = os.path.join(root, file)
            trajectory_df:gpd.GeoDataFrame = get_trajectory_df_from_txt(file_path=file_path)
            
            file_name = file.split('/')[-1]
            file_folder = root.split('/')[-1]
            vessel_folder = trajectory_df.iloc[0].ship_type.replace('/', '_')
            vessel_folder_path = f'{folder_path}/{vessel_folder}'
            new_file_path = os.path.join(vessel_folder_path, file_folder)
            
            os.makedirs(new_file_path, exist_ok=True)  # Create the folder if it doesn't exist
        
            if len(trajectory_df) >= 2:
                # calculate bearing between consecutive points
                
                trajectory_df['time_diff'] = trajectory_df.timestamp.diff().fillna(0)
                
                gdf_next = trajectory_df.shift(-1)
                trajectory_df, gdf_next = utils.get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
                
                bearing_df = utils.calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
                trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

                trajectory_df['dist'] = utils.get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
     
                # # Calculate speed for points with subsequent points available
                speeds = trajectory_df['dist'] / trajectory_df['time_diff']
                speeds.fillna(0, inplace=True)
                
                trajectory_df['speed'] = speeds
                
                mask = (trajectory_df['speed'] >= 3) & (trajectory_df['time_diff'] >= 15) | \
                        (trajectory_df['speed'] < 3) & (trajectory_df['time_diff'] >= 210)

                # Ensure the first and last row are always included
                mask.iloc[0] = True # Ensure the first row is always included
                mask.iloc[-1] = True # Ensure the last row is always included
                
                # Filter DataFrame
                sparse_trajectory_df = trajectory_df[mask]                
                                
                sparse_trajectory_df[['latitude', 'longitude', 'timestamp', 'cog', 'draught', 'ship_type', 'speed']].to_csv(f'{new_file_path}/{file_name}', sep=',', index=True, header=True, mode='w')     
                
                total_number_of_points += len(trajectory_df)
                removed += len(trajectory_df) - len(sparse_trajectory_df)
            
    finished_time = t.perf_counter() - initial_time
    removed_avg = removed/num_files
    logging.info(f'Removed on avg. pr trajectory: {removed_avg}. Removed in total: {removed}/{total_number_of_points}. Elapsed time: {finished_time:0.4f} seconds')   

#extract_trajectories_from_csv_files()
#filter_original_trajectories(sog_threshold=0.0)
sparcify_realisticly_original_trajectories(INPUT_ALL_TEST_FOLDER)
for root, folder, files in os.walk(ORIGINAL_FOLDER):
    is_empty = (len(files) == 0)
    if (is_empty):
        print(':((')


#sparcify_original_trajectories_for_folder(folder_path=INPUT_FOLDER_ALL)