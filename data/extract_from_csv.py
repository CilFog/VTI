import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sys import stdout
from shapely import wkb
from create_tractories import create_trajectories_from_df
from logs.logging import setup_logger

AIS_CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'ais_csv')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
GTI_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../GTI/data/ais')
TRIMPUTE_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../TrImpute/datasets/input/ais_csv')
TEST_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'test_data')
TXT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'txtfiles')
HARBORS_FILE = os.path.join(os.path.dirname(__file__), 'harbors.csv')
CSV_EXTRACT_FILE_LOG = 'ais_extraction_log.txt'

logging = setup_logger(CSV_EXTRACT_FILE_LOG)
    
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
    
    try:
        for file in files_to_download:
            extract_csv_file(file_name=file)
            downloaded += 1
            stdout.write(f'\rDownloaded {file}.  Completed ({downloaded}/{len(files_to_download)})')
            stdout.flush()
        logging.info('\nCompleted extracting csv files')
    except Exception as e:
        logging.error(f'Failed to extract csv "{file}" file with error: {repr(e)}')

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
    #create_csv_file_for_mmsis(file_path=os.path.join(AIS_CSV_FOLDER,'aisdk-2024-02-11.csv')) #specify in method which mmsi
    file_names = os.listdir(TEST_DATA_FOLDER)
    #file_names = os.listdir(AIS_CSV_FOLDER)
    completed:int = 0
    
    logging.info(f'Began extracting trajectories from {len(file_names)} csv files')
    
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        logging.info(f'Currently extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')        
        df:gpd.GeoDataFrame = cleanse_csv_file_and_convert_to_df(os.path.join(TEST_DATA_FOLDER, file_name))
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
    
    # Grouping by the columns 'imo', 'name', 'length', 'width', and 'ship_type'
    # and filling missing values within each group with the first non-null value
    df[['imo', 'name', 'length', 'width', 'ship_type']] = df.groupby('mmsi')[['imo', 'name', 'length', 'width', 'ship_type']].transform(lambda x: x.ffill())

    # Filling any remaining missing values with the last non-null value
    df[['imo', 'name', 'length', 'width', 'ship_type']] = df.groupby('mmsi')[['imo', 'name', 'length', 'width', 'ship_type']].transform(lambda x: x.bfill())
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    df = df.drop(columns=['index'], errors='ignore')
    df = df.to_crs(epsg="3857") # to calculate in meters
    
    return df

def create_csv_file_for_mmsis(file_path: str):
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
    
    mmsis = [211190000, 210524000, 210853000, 210549000, 209536000, 210332000]
    
    df = pd.read_csv(file_path, na_values=['Unknown','Undefined'], dtype=types) #, nrows=1000000)    
    
    if (df.empty):
        logging.warning('No data found in csv')

    df_filtered = df[df['MMSI'].isin(mmsis)]

    if df_filtered.empty:
        logging.warning('Could not find MMSI in the csv. Nothing created')
    else:
        for mmsi in mmsis:
            df_mmsi = df_filtered[df_filtered['MMSI'] == mmsi]
            df_mmsi.reset_index().to_csv(f'{TEST_DATA_FOLDER}/{mmsi}.csv', index=False)  # Set index=False if you don't want to write row numbers
            df_mmsi.reset_index().to_csv(f'{TXT_DATA_FOLDER}/{mmsi}.txt', index=False) 
            logging.info(f'csv for {mmsi} created')
        
def create_trajectories_files(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return gdf
    
    harbors_df = extract_harbors_df()
    
    for mmsi, trajectory_df in gdf.sort_values('timestamp').groupby('mmsi'):
        logging.info(f'Creating trajectories for mmsi {mmsi}')
                
        #trajectory_df.reset_index().to_csv(f'{TEST_DATA_FOLDER}/{mmsi}.csv', index=False)  # Set index=False if you don't want to write row numbers
        #trajectory_df.reset_index().to_csv(f'{TXT_DATA_FOLDER}/{mmsi}.txt', index=False) 
        
        sub_trajectories_df = create_trajectories_from_df(harbors_df=harbors_df, trajectory_df=trajectory_df) 
        
        if not sub_trajectories_df.empty:
            write_to_input_folder(sub_trajectories_df)
            #write_gti_input_trajectories(sub_trajectories_df)
            #write_TrImpute_input_trajectories(sub_trajectories_df)

def extract_harbors_df() -> gpd.GeoDataFrame:    
    df = pd.read_csv(HARBORS_FILE, na_values=['Unknown','Undefined']) 
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(bytes.fromhex(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.drop(columns=['index'], errors='ignore')
    gdf = gdf.to_crs(epsg="3857") # to calculate in meters
    
    return gdf

def write_gti_input_trajectories(gdf: gpd.GeoDataFrame):    
    if (gdf.empty):
        return
    
    os.makedirs(GTI_INPUT_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
    
    current_files = os.listdir(GTI_INPUT_FOLDER)
    trip_id = len(current_files)

    for _, trajectories in gdf.groupby(['sub_trajectory_id']):        
        file_path = os.path.join(GTI_INPUT_FOLDER, f"trip_{trip_id}.txt")
        trajectories[['latitude', 'longitude', 'timestamp']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=False, mode='w')
        trip_id += 1

def write_TrImpute_input_trajectories(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return
    
    os.makedirs(TRIMPUTE_INPUT_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
    
    current_files = os.listdir(TEST_DATA_FOLDER)

    trip_id = len(current_files)
    
    for _, sub_trajectories in gdf.groupby(['sub_trajectory_id']):  
        if len(sub_trajectories) < 2:
            continue      
        file_path = os.path.join(TRIMPUTE_INPUT_FOLDER, f"trip_{trip_id}.txt")
        sub_trajectories[['latitude', 'longitude', 'timestamp']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=False, mode='w')
        trip_id += 1
        
def write_to_input_folder(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return
    
    for _, sub_trajectories in gdf.groupby('sub_trajectory_id'):
        if len(sub_trajectories) < 2:
            continue
        
        folder_path = os.path.join(INPUT_FOLDER, str(sub_trajectories.iloc[0].mmsi))
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        
        dt_object = dt.datetime.utcfromtimestamp(sub_trajectories.iloc[0].timestamp)
        dt_str = dt_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_')
            
        file_path = os.path.join(folder_path, f'{dt_str}.txt')        
        sub_trajectories[['latitude', 'longitude', 'timestamp', 'sog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=True, mode='w')
#get_csv_files_in_interval("2024-02-10::2024-02-12")
extract_trajectories_from_csv_files()