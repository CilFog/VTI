import os
import tarfile
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sys import stdout
from logs.logging import setup_logger

AIS_CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'ais_csv')
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'test_data')
TXT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'txtfiles')
CSV_EXTRACT_FILE_LOG = 'get_csv_files_log.txt'

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

    path_to_compressed_file = AIS_CSV_FOLDER + file_name
    os.makedirs(AIS_CSV_FOLDER, exist_ok=True)

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
                zip_ref.extractall(path=AIS_CSV_FOLDER)
        elif ".rar" in path_to_compressed_file:
            with tarfile.RarFile(path_to_compressed_file) as rar_ref:
                rar_ref.extractall(path=AIS_CSV_FOLDER)
        else:
            logging.error(f'File {file_name} must either be of type .zip or .rar. Not extracted')
            
        os.remove(path_to_compressed_file)

    except Exception as e:
        logging.exception(f'Failed with error: {e}')
        quit()

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
    
    mmsis = [210388000, 211190000, 210524000, 210853000, 210549000, 209536000, 210332000]
    
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
            # df_mmsi.reset_index().to_csv(f'{TXT_DATA_FOLDER}/{mmsi}.txt', index=False) 
            logging.info(f'csv for {mmsi} created')