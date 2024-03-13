import os
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
from shapely import wkb
from split_tractories import split_trajectories_from_df
from logs.logging import setup_logger

AIS_CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'ais_csv')
ORIGINAL_FOLDER = os.path.join(os.path.dirname(__file__), 'original')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input_graph')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
TEST_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'test_data')
TXT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'txtfiles')
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
            
        os.remove(file_path)
    
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

def write_60_minute_split_trajectory(file_name: str, folder:str, sub_trajectories: gpd.GeoDataFrame):
    df_hour = sub_trajectories.copy()
    df_hour['hour'] = df_hour['timestamp'] // 3600

    # Step 2: Group by the 'hour' column and select the first row of each group
    df_hour = df_hour.groupby('hour').first().reset_index(drop=True)
    
    folder_name = f'{folder}/60'
    folder_path = os.path.join(INPUT_FOLDER, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    file_path = os.path.join(folder_path, file_name)
    
    write_trajectories(file_path, df_hour)

def write_30_minute_split_trajectory(file_name: str, folder: str, sub_trajectories: gpd.GeoDataFrame):
    df_thirty = sub_trajectories.copy()
    df_thirty['30min_interval'] = df_thirty['timestamp'] // 1800

    # Step 2: Group by the 'hour' column and select the first row of each group
    df_thirty = df_thirty.groupby('30min_interval').first().reset_index(drop=True)
    
    folder_name = f'{folder}/30'
    folder_path = os.path.join(df_thirty, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    write_trajectories(file_name, df_thirty)

def write_15_minute_split_trajectory(file_name: str, sub_trajectories: gpd.GeoDataFrame):
    return

def write_trajectories(file_path:str, sub_trajectories: gpd.GeoDataFrame):
    sub_trajectories[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=True, mode='w')
