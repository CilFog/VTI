import os
import tarfile
import zipfile
import requests
import logging 
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sys import stdout

CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'csv')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
GTI_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../GTI/data/ais')
TRIMPUTE_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../TrImpute/datasets/input/ais_csv')
TEST_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'test_data')

logging.basicConfig(level=logging.INFO)
    
def get_csv_files_in_interval(interval: str):
    """
    Downloads and processes all the available ais data in the given interval.
    :param interval: The interval (date) to download and process.
    """
    dates = interval.split('::')
    csv_files_on_server = connect_to_to_ais_web_server_and_get_data()
    begin_index = None
    end_index = None

    for csv_file in csv_files_on_server:
        if begin_index is None:
            if dates[0] in csv_file:
                begin_index = csv_files_on_server.index(csv_file)
                continue
        if end_index is None:
            if dates[1] in csv_file:
                end_index = csv_files_on_server.index(csv_file)

    files_to_download = csv_files_on_server[begin_index:end_index]

    file: str
    downloaded:int = 0
    
    logging.info('Began extracting csv files')
    
    for file in files_to_download:
        extract_csv_file(file_name=file)
        downloaded += 1
        stdout.write(f'\rDownloaded {file}.  Completed ({downloaded}/{len(files_to_download)})')
        stdout.flush()
    logging.info('\nCompleted extracting csv files')

def connect_to_to_ais_web_server_and_get_data():
    """
    Connects to the ais web server and gets the .csv files (ais data) located there.
    :param logging: A logging for loggin warning/errors
    :return: A list with the names of the .zip/.rar files available for download on the web server. Example: 'aisdk-2022-01-01.zip'
    """
    logging.info('Began retrievel of data from https://web.ais.dk/aisdata/')
    
    html = urlopen("https://web.ais.dk/aisdata/")
    
    try:    
        soup = BeautifulSoup(html, 'html.parser')
        
        all_links = soup.find_all('a', href=lambda href: True and 'aisdk' in href)
        all_links_as_string = [link.string for link in all_links]

        logging.info(f'\nCompleted fetching {len(all_links_as_string)} ais download links')
        return all_links_as_string
    except Exception as e:
        logging.error('Fetching AIS data failed with: %s', repr(e))

def extract_csv_file(file_name: str):
    """
    Downloads the given file, runs it through the pipeline and adds the file to the log.
    :param file_name: The file to be downloaded, cleansed and inserted
    """
    download_file_from_ais_web_server(file_name)

    try:
        if ".zip" in file_name: 
            file_name = file_name.replace('.zip', '.csv')
        else:
            file_name = file_name.replace('.rar', '.csv')
        # df = cleanse_csv_file_and_convert_to_df(file_name=file_name)
    except Exception as e:
        logging.error(f'Failed to unzip file {file_name} with error message: {repr(e)}')

def download_file_from_ais_web_server(file_name: str):
    """
    Downloads a specified file from the webserver into the CSV_FILES_FOLDER.
    It will also unzip it, as well as delete the compressed file afterwards.
    :param file_name: The file to be downloaded. Example 'aisdk-2022-01-01.zip'
    """
    download_result = requests.get('https://web.ais.dk/aisdata/' + file_name, allow_redirects=True)
    download_result.raise_for_status()

    path_to_compressed_file = INPUT_FOLDER + file_name

    try:
        f = open(path_to_compressed_file,'wb')
        f.write(download_result.content)
    except Exception as e:
        logging.exception(f'Failed to retrieve file from ais web server, with messega: {repr(e)}')
    finally:
        f.close()
    
    try:
        if ".zip" in path_to_compressed_file: 
            with zipfile.ZipFile(path_to_compressed_file, 'r') as zip_ref:
                zip_ref.extractall(INPUT_FOLDER)
        elif ".rar" in path_to_compressed_file:
            with tarfile.RarFile(path_to_compressed_file) as rar_ref:
                rar_ref.extractall(path=INPUT_FOLDER)
        else:
            logging.error(f'File {file_name} must either be of type .zip or .rar. Not extracted')
            
        os.remove(path_to_compressed_file)

    except Exception as e:
        logging.exception(f'Failed with error: {e}')
        quit()

def extract_trajectories_from_csv_files():
    
    #create_csv_file_for_mmsi(file_name=os.path.join('../csv','aisdk-2024-02-11.csv'), mmsi=219423000)
    file_names = os.listdir(TEST_DATA_FOLDER)
    #file_names = os.listdir(INPUT_FOLDER)
    existing_trips = os.listdir(OUTPUT_FOLDER)
    trajectory_id = 0
    
    if len(existing_trips) != 0:
        trajectory_id = int(file_name.split('_')[1]) + 1

    completed:int = 0
    
    logging.info(f'Began extracting trajectories from {len(file_names)} csv files')
    
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        
        logging.info(f'Currently extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')        
        
        df:gpd.GeoDataFrame = cleanse_csv_file_and_convert_to_df(file_name)
        
        completed +=1
        
        logging.info(f'Finished extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')        
        logging.info(f'Began crating trajectories for file: {file_name}')

        if (not df.empty):
            trajectory_id = create_trajectories(trajectory_id, df)
        else:
            logging.warning(f'No data was extracted from {file_name}')
    
    logging.info('Finished creating trajecatories. Terminating')

def cleanse_csv_file_and_convert_to_df(file_name: str):
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
    
    #df = pd.read_csv(str.format("{0}/{1}", INPUT_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types)#, nrows=1000000)    
    df = pd.read_csv(str.format("{0}/{1}", TEST_DATA_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types)#, nrows=1000000)    
    
    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type', 'Destination', 'Type of position fixing device',
                  'Callsign', 'Name'],axis=1, errors='ignore')
        
    # Remove all the rows which does not satisfy our conditions
    df = df[
            (df["Type of mobile"] != "Class B") &
            (df["MMSI"].notna()) &
            (df["MMSI"].notnull()) &
            (df['# Timestamp'].notnull()) &
            (df['Latitude'] >=53.5) & (df['Latitude'] <=58.5) &
            (df['Longitude'] >= 3.2) & (df['Longitude'] <=16.5) &
            (df['SOG'] >= 0.1) & (df['SOG'] <=102)
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
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    df = df.drop(columns=['index'], errors='ignore')
    df = df.to_crs(epsg="3857") # to calculate in meters
    
    return df

def create_csv_file_for_mmsi(file_name: str, mmsi: int):
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

    df = pd.read_csv(str.format("{0}/{1}", INPUT_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types) #, nrows=1000000)    
    
    if (df.empty):
        logging.warning('No data found in csv')
    
    df = df[df['MMSI'] == mmsi]
    
    if (df.empty):
        logging.warning('Could not find mmsi in the csv. Nothing created')
    
    df.to_csv(f'{TEST_DATA_FOLDER}/{str(mmsi)}.csv', index=False)  # Set index=False if you don't want to write row numbers
    
    logging.info(f'csv for {mmsi} created in {TEST_DATA_FOLDER}')
  
def create_trajectories(trajectory_id: int, gdf: gpd.GeoDataFrame) -> int:
    if (gdf.empty):
        return gdf
    
    for mmsi, locations in gdf.sort_values('timestamp').groupby('mmsi'):
        logging.info(f'Creating trajectories for mmsi {mmsi}')
        sub_trajectories_df = make_sub_trajectories(trajectory_id=trajectory_id, sorted_locations_gdf=locations) 
        
        write_gti_input_trajectories(sub_trajectories_df)
        write_TrImpute_input_trajectories(sub_trajectories_df)
        
        trajectory_id += 1
    
    return trajectory_id

def make_sub_trajectories(trajectory_id: int, sorted_locations_gdf: gpd.GeoDataFrame):
    if sorted_locations_gdf.empty:
        return sorted_locations_gdf
        

    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True)
    sorted_locations_gdf = order_by_diff_vessels(sorted_locations_gdf)
    sorted_locations_gdf = sorted_locations_gdf.drop_duplicates()
    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True) # to ensure indexes are still fine
    
    radius_threshold = 5 # meters, diameter is 10
    sub_trajectories = []
    sub_trajectory_id = 1
    moving = True
    
    for vessel_id, locations_df in sorted_locations_gdf.groupby('vessel_id'):
        current_sub_trajectory = []
        consecutive_points_within_radius = []

        for index, row in locations_df[1:].iterrows():
            current_location = row
            last_location = locations_df.iloc[index - 1]
            distance = current_location.geometry.distance(last_location.geometry)
            
            if (consecutive_points_within_radius):
                distance = consecutive_points_within_radius[0].geometry.distance(current_location.geometry)

            if moving:  # If moving
                if (current_location.geometry == last_location.geometry and current_location.sog == 0.0):  # Not moving
                    moving = False
                    consecutive_points_within_radius.append(current_location)
                    continue

                if distance <= radius_threshold:  # Check if within radius threshold
                    consecutive_points_within_radius.append(current_location)
                    if len(consecutive_points_within_radius) >= 5:
                        moving = False
                else:
                    if consecutive_points_within_radius:  # If non-empty
                        current_sub_trajectory.extend(consecutive_points_within_radius)
                        consecutive_points_within_radius = []  # Reset
                    current_sub_trajectory.append(current_location)
            else:  # If not moving
                if distance > radius_threshold:
                    # create sub trjaectory df
                    if current_sub_trajectory: 
                        sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
                        sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
                        sub_trajectories.append(sub_trajectory_df)
                        sub_trajectory_id += 1
                    print(f'made trajectry {sub_trajectory_id}')    
                    # reset for next sub trajectory
                    current_sub_trajectory = []  # Reset
                    consecutive_points_within_radius = []  # Reset
                    current_sub_trajectory.append(current_location)
                    moving = True
                else:
                    continue

        if current_sub_trajectory:  # Append the last sub-trajectory if not empty
            sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
            sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
            sub_trajectories.append(sub_trajectory_df)
                      
    result_sub_trajectory_df = pd.concat(sub_trajectories, ignore_index=True)
    result_sub_trajectory_df['trajectory_id'] = trajectory_id 

    return result_sub_trajectory_df

def order_by_diff_vessels(sorted_locations_df: gpd.GeoDataFrame):
    sorted_locations_df.fillna({'imo': -1, 'ship_type': 'None', 'width': -1, 'length': -1}, inplace=True)
    sorted_locations_df['vessel_id'] = sorted_locations_df.groupby(['imo', 'ship_type', 'width', 'length']).ngroup() 

    return sorted_locations_df.dropna()
    # """
    # Check if the next point is within 10 km of the current point
    # """   
    distance = current_location.geometry.distance(next_location.geometry) # distance is in meters
    return distance <= 10000
    
def write_gti_input_trajectories(gdf: gpd.GeoDataFrame):
    # Group by 'trajectory_id' and 'trajectory_sub_id' and iterate over each group
    
    if (gdf.empty):
        return
    
    for (trajectory_id, trajectory_sub_id), trajectories in gdf.groupby(['trajectory_id', 'sub_trajectory_id']):        
        file_path = os.path.join(GTI_INPUT_FOLDER, f"trip_{trajectory_id}_{trajectory_sub_id}.txt")
        print(file_path)
        with open(file_path, 'w') as file:
            # Iterate over rows in the group
            for idx, row in trajectories.reset_index().iterrows():
                # Write timestamp, latitude, and longitude to the file
                file.write(f"{idx},{row['latitude']},{row['longitude']},{row['timestamp']}\n")

def write_TrImpute_input_trajectories(gdf: gpd.GeoDataFrame):
    # Group by 'trajectory_id' and 'trajectory_sub_id' and iterate over each group
    if (gdf.empty):
        return
        
    for (trajectory_id, trajectory_sub_id), trajectories in gdf.groupby(['trajectory_id', 'sub_trajectory_id']):        
        file_path = os.path.join(TRIMPUTE_INPUT_FOLDER, f"trip_{trajectory_id}{trajectory_sub_id}.txt")
        trajectories[['latitude', 'longitude', 'timestamp']].to_csv(file_path, index=True)
        
def write_to_output_folder(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return
    gdf.sort_values('timestamp')
    for mmsi, trajectories in gdf.groupby('mmsi'):        
        file_name = os.path.join(OUTPUT_FOLDER, f"{mmsi}.txt")
        with open(file_name, 'w') as file:
            # Iterate over rows in the group
            for idx, row in trajectories.reset_index().iterrows():
                # Write timestamp, latitude, and longitude to the file
                file.write(f"{idx},{row['mmsi']},{row['latitude']},{row['longitude']},{row['timestamp']}\n")

#get_csv_files_in_interval("2024-02-10::2024-02-12")
extract_trajectories_from_csv_files()