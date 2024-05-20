import os
import sys
import shutil
import random
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List
from shapely.geometry import box
from multiprocessing import freeze_support
from data.logs.logging import setup_logger
from data.stats.statistics import Sparse_Statistics
from .classes import brunsbuettel_to_kiel_polygon, aalborg_harbor_to_kattegat_bbox, doggersbank_to_lemvig_bbox, skagens_harbor_bbox
from .sparcify_methods import sparcify_trajectories_realisticly, sparcify_large_meter_gap_by_threshold, sparcify_trajectories_with_meters_gaps_by_treshold, get_trajectory_df_from_txt, check_if_trajectory_is_dense

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

STATS_FOLDER = os.path.join(DATA_FOLDER, 'stats')
STATS_INPUT_IMPUTATION = os.path.join(STATS_FOLDER, 'input_imputation')
STATS_TEST = os.path.join(STATS_INPUT_IMPUTATION, 'test')
STATS_TEST_ALL = os.path.join(STATS_TEST, 'all')
STATES_TEST_AREA = os.path.join(STATS_TEST, 'area')
STATS_VALIDATION = os.path.join(STATS_INPUT_IMPUTATION, 'validation')
STATS_VALIDATION_ALL = os.path.join(STATS_VALIDATION, 'all')
STATS_VALIDATIOM_AREA = os.path.join(STATS_VALIDATION, 'area')
SPARCIFY_LOG = 'sparcify_log.txt'

logging = setup_logger(name=SPARCIFY_LOG, log_file=SPARCIFY_LOG)

def write_trajectories_for_area(file:str, outputfolder: str, output_json:str):
    aalborg_harbor_to_kattegat_path = os.path.join(outputfolder, 'aalborg_harbor_to_kattegat')
    skagen_harbor_path = os.path.join(outputfolder, 'skagen_harbor')
    
    outputfolders = [
        (aalborg_harbor_to_kattegat_path, aalborg_harbor_to_kattegat_bbox),
        (skagen_harbor_path, skagens_harbor_bbox)]
    
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
    
    outputfolders = [
        (aalborg_harbor_to_kattegat_path, aalborg_harbor_to_kattegat_bbox),
        (skagen_harbor_path, skagens_harbor_bbox)]
    
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

threshold_values = [500, 1000, 2000, 4000, 8000]
# test_all_threshold = os.path.join(STATS_TEST_ALL, 'all_threshold.json')
# test_area_threshold = os.path.join(STATES_TEST_AREA, 'area_threshold.json')
# test_all_realistic = os.path.join(STATS_TEST_ALL, 'all_realistic.json')
# test_area_realistic = os.path.join(STATES_TEST_AREA, 'area_realistic.json')
validation_all_threshold = os.path.join(STATS_VALIDATION_ALL, 'all_threshold.json')
validation_area_threshold = os.path.join(STATS_VALIDATIOM_AREA, 'area_threshold.json')
validation_all_realistic = os.path.join(STATS_VALIDATION_ALL, 'all_realistic.json')
validation_area_realistic = os.path.join(STATS_VALIDATIOM_AREA, 'area_realistic.json')

# os.makedirs(STATS_TEST_ALL, exist_ok=True)
# os.makedirs(STATES_TEST_AREA, exist_ok=True)
os.makedirs(STATS_VALIDATION_ALL, exist_ok=True)
os.makedirs(STATS_VALIDATIOM_AREA, exist_ok=True)
stats = Sparse_Statistics()

# test_files = list(Path(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER).rglob('*.txt')) # List all files in the directory recursively
# test_files = [str(path) for path in test_files] # Convert Path objects to strings

# random_test_files_to_move = random.sample(test_files, 1000)

# print('making data for test')
# for file in random_test_files_to_move:
#     for threshold in threshold_values:
#         write_trajectories_for_all_with_threshold(file, INPUT_TEST_SPARSED_ALL_FOLDER, threshold=threshold, output_json=test_all_threshold)
#         write_trajectories_for_area_with_threshold(file, INPUT_TEST_SPARSED_AREA_FOLDER, threshold=threshold, output_json=test_area_threshold)

#     write_trajectories_for_all(file, INPUT_TEST_SPARSED_ALL_FOLDER, output_json=test_all_realistic)
#     write_trajectories_for_area(file, INPUT_TEST_SPARSED_AREA_FOLDER, output_json=test_area_realistic)

# print('making stats for test with')
# stats.make_statistics_with_threshold(test_all_threshold)
# stats.make_statistics_with_threshold(test_area_threshold)
# stats.make_statistics_no_threshold(test_all_realistic)
# stats.make_statistics_no_threshold(test_area_realistic)

paths = []
print('gettings vessels folders')
vessel_folders = Path(INPUT_VALIDATION_DATA_ORIGINAL_FOLDER).iterdir() if folder.is_dir()]
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
            write_trajectories_for_all_with_threshold(file, INPUT_VALIDATION_SPARSED_ALL_FOLDER, threshold=threshold, output_json=validation_all_threshold)
            write_trajectories_for_area_with_threshold(file, INPUT_VALIDATION_SPARSED_AREA_FOLDER, threshold=threshold, output_json=validation_area_threshold)

        write_trajectories_for_all(file, INPUT_VALIDATION_SPARSED_ALL_FOLDER, output_json=validation_all_realistic)
        write_trajectories_for_area(file, INPUT_VALIDATION_SPARSED_AREA_FOLDER, output_json=validation_area_realistic)

print('making stats for validation')
stats.make_statistics_with_threshold(validation_all_threshold)
stats.make_statistics_with_threshold(validation_area_threshold)
stats.make_statistics_no_threshold(validation_all_realistic)
stats.make_statistics_no_threshold(validation_area_realistic)