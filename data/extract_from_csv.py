import os
import tarfile
import zipfile
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import requests
import pandas as pd
import datetime as dt

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
GTI_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../GTI/data/ais')


def connect_to_to_ais_web_server_and_get_data():
    """
    Connects to the ais web server and gets the .csv files (ais data) located there.
    :param logger: A logger for loggin warning/errors
    :return: A list with the names of the .zip/.rar files available for download on the web server. Example: 'aisdk-2022-01-01.zip'
    """
    html = urlopen("https://web.ais.dk/aisdata/")
    

    soup = BeautifulSoup(html, 'html.parser')
    results = []

    for link in soup.find_all('a', href = True):
        if "aisdk" in link.string:
            results.append(link.string)

    return results

def download_interval(interval: str):
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
    for file in files_to_download:
        download_cleanse_insert(file_name=file)

def download_cleanse_insert(file_name: str):
    """
    Downloads the given file, runs it through the pipeline and adds the file to the log.
    :param file_name: The file to be downloaded, cleansed and inserted
    """
    download_file_from_ais_web_server(file_name)
    files_to_insert = []

    if len(files_to_insert) <= 0:
        files_to_insert.append(file_name)
    
    for file in files_to_insert:
        file_name = file
        if ".zip" in file: 
            file_name = file.replace('.zip', '.csv')
        else:
            file_name = file.replace('.rar', '.csv')
        df = cleanse_csv_file_and_convert_to_df(file_name=file_name)
        print(df.head())
        #partition_trips_and_insert(file, df)

def download_file_from_ais_web_server(file_name: str):
    """
    Downloads a specified file from the webserver into the CSV_FILES_FOLDER.
    It will also unzip it, as well as delete the compressed file afterwards.
    :param file_name: The file to be downloaded. Example 'aisdk-2022-01-01.zip'
    :param logger: A logger to log warnings/errors.
    """
    download_result = requests.get("https://web.ais.dk/aisdata/" + file_name, allow_redirects=True)
    download_result.raise_for_status()


    path_to_compressed_file = INPUT_FOLDER + file_name

    try:
        f = open(path_to_compressed_file,'wb')
        f.write(download_result.content)
    except Exception as e:
        print("Error")
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
            print("Error")
            
        os.remove(path_to_compressed_file)

    except Exception as err:
        print("Error")
        quit()

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
        'Data source type': str,
    }

    df = pd.read_csv(str.format("{0}/{1}", INPUT_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types) #, nrows=1000000    
    
    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type'],axis=1, errors='ignore')
    
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format="%d/%m/%Y %H:%M:%S")
    df['# Timestamp'].astype('int64')//1e9
    df['# Timestamp'] = (df['# Timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()    

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

    df = df.drop(columns=['index','type_of_mobile','type_of_position_fixing_device', 'width', 'length', 'name', 'callsign','imo', 'destination','navigational_status', 'rot'], errors='ignore')
        
    return df

def create_trajectory_gti(df:pd.DataFrame, trip_id):
    # Group by 'mmsi' and iterate over each group
    for mmsi, group_df in df.groupby('mmsi'):
        file_name = os.path.join(GTI_OUTPUT_FOLDER, f"trip{trip_id}.txt")
        with open(file_name, 'w') as file:
            # Iterate over rows in the group
            for idx, row in group_df.reset_index().iterrows():
                # Write timestamp, latitude, and longitude to the file
                file.write(f"{idx},{row['latitude']},{row['longitude']},{row['timestamp']}\n")
        trip_id += 1

def create_gti_txt_trajectories():
    filenames = os.listdir(INPUT_FOLDER)
    print(filenames)
    trip_id = 0
    for file_index in range(len(filenames)):
        file_name = filenames[file_index]
        print('creating trajectories for ' + file_name)
        df = cleanse_csv_file_and_convert_to_df(file_name)
        create_trajectory_gti(trip_id)
        print('finished trajectories for ' + file_name)

        trip_id += df.__len__()

#download_interval("2024-02-10::2024-02-12")

#cleanse_csv_file_and_convert_to_df("aisdk-2024-02-12.csv")
create_gti_txt_trajectories()
