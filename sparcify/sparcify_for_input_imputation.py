import os
import sys
import random
import shutil
import time as t
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Callable
from shapely.geometry import Polygon, box
from multiprocessing import freeze_support
from data.logs.logging import setup_logger
from concurrent.futures import ProcessPoolExecutor
from .classes import SparsifyResult, brunsbuettel_to_kiel_polygon, aalborg_harbor_to_kattegat_bbox, doggersbank_to_lemvig_bbox, skagens_harbor_bbox
from .sparcify_methods import sparcify_realisticly_strict_trajectories, sparcify_trajectories_realisticly, sparcify_large_time_gap_with_threshold_percentage, sparcify_trajectories_randomly_using_threshold, get_trajectory_df_from_txt, check_if_trajectory_is_dense

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
SPARCIFY_LOG = 'sparcify_log.txt'

logging = setup_logger(name=SPARCIFY_LOG, log_file=SPARCIFY_LOG)

def write_trajectories_for_area(input_folder:str, output_folder: str):
    # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        brunsbuettel_to_kiel_path = os.path.join(output_folder, 'brunsbuettel_to_kiel')

        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=brunsbuettel_to_kiel_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=brunsbuettel_to_kiel_polygon)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=brunsbuettel_to_kiel_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=brunsbuettel_to_kiel_polygon)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=brunsbuettel_to_kiel_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=brunsbuettel_to_kiel_polygon)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=brunsbuettel_to_kiel_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=brunsbuettel_to_kiel_polygon)

        aalborg_harbor_to_kattegat_path = os.path.join(output_folder, 'aalborg_harbor_to_kattegat')

        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=aalborg_harbor_to_kattegat_bbox)

        doggersbank_to_lemvig_path = os.path.join(output_folder, 'doggersbank_to_lemvig')

        sparcify_trajectories_with_action_for_folder(input_folder=input_folder,folder_path=doggersbank_to_lemvig_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=doggersbank_to_lemvig_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=doggersbank_to_lemvig_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=doggersbank_to_lemvig_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)

        skagen_harbor_path = os.path.join(output_folder, 'skagen_harbor')

        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=skagen_harbor_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=skagen_harbor_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=skagen_harbor_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=skagen_harbor_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)

def write_trajectories_for_all(input_folder: str, output_folder:str):

    # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=output_folder, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=output_folder, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=output_folder, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(input_folder=input_folder, folder_path=output_folder, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=None)

def sparcify_trajectories_with_action_for_folder(
    input_folder: str,
    folder_path: str, 
    action: Callable[[str, float, Polygon], SparsifyResult], 
    threshold: float = 0.0,  # Example default value for threshold
    boundary_box: Polygon = None  # Default None, assuming Polygon is from Shapely or a similar library
):
    initial_time = t.perf_counter()
    total_reduced_points = 0
    total_number_of_points = 0
    total_trajectories = 0
    reduced_avg = 0

    logging.info(f'Began sparcifying trajectories with {str(action)}')

    # List all files in the directory recursively
    file_paths = list(Path(input_folder).rglob('*.txt'))

    # Convert Path objects to strings 
    file_paths = [str(path) for path in file_paths]

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(action, file_paths, [folder_path] * len(file_paths), [threshold] * len(file_paths), [boundary_box] * len(file_paths))        
        for result in results:
            total_reduced_points += result.reduced_points
            total_number_of_points += result.number_of_points
            total_trajectories += 1 if result.trajectory_was_used else 0
    
    if total_trajectories == 0:
        print('No trajectories were used')
    else:
        reduced_avg = total_reduced_points/total_trajectories
    
    finished_time = t.perf_counter() - initial_time
    logging.info(f'Reduced on avg. pr trajectory: {reduced_avg} for {total_trajectories} trajectories. Reduced points in total: {total_reduced_points}/{total_number_of_points}. Elapsed time: {finished_time:0.4f} seconds')   

def move_random_files_to_test_and_validation(percentage=0.1):
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'
    directories_with_moved_files = set()
    all_files = list(Path(INPUT_GRAPH_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    all_files = [str(path) for path in all_files] # Convert Path objects to strings

    num_files_to_move_to_validation = int(len(all_files) * percentage)
    not_dense_files:list[str] = []
    files_moved:list[str] = []

    # for test
    try:
        cells_paths = [f'input_graph_cells{os_path_split}9_9', f'input_graph_cells{os_path_split}9_10', f'input_graph_cells{os_path_split}9_11', f'input_graph_cells{os_path_split}10_9', f'input_graph_cells{os_path_split}10_10', f'input_graph_cells{os_path_split}10_11', f'input_graph_cells{os_path_split}11_9', f'input_graph_cells{os_path_split}11_10', f'input_graph_cells{os_path_split}11_11']
        cells = ['9_9', '9_10', '9_11', '10_9', '10_10', '10_11', '11_9', '11_10', '11_11']
        
        # get files (vessel samples) within all cells
        cell_files = list(Path(INPUT_GRAPH_CELLS_FOLDER).rglob('*.txt')) # List all files in the directory recursively
        cell_files = [str(path) for path in cell_files]
        
        # filter all files not within the specified cells paths
        cell_files = [path for path in cell_files if any(cell in path for cell in cells_paths)]
        
        # read cells in txt
        cells_df = pd.read_csv(CELL_TXT)

        # filter cells not in the cells array
        filtered_cells_df = cells_df[cells_df['cell_id'].isin(cells)]
        filtered_cells_df['geometry'] = filtered_cells_df.apply(lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']), axis=1)
        cells_gdf = gpd.GeoDataFrame(filtered_cells_df, geometry='geometry', crs="EPSG:4326")

        # look for files (whole trajectories) in input graph that we know intersects with the cells
        all_files_with_cells = [path for path in all_files if any(os.path.basename(cell) in path for cell in cell_files)]
        
        # calculate the number of files to move to test given the percentage
        num_files_to_move_to_test = int(len(all_files_with_cells) * percentage)
        logging.info(f'Began moving test files {num_files_to_move_to_test}')
        
        # select random files
        random_files_to_move = random.sample(all_files_with_cells, num_files_to_move_to_test)
        
        num_files_moved_to_test = 0

        while(num_files_moved_to_test < num_files_to_move_to_test):
            for filepath in random_files_to_move:
                # check we are not done
                if (num_files_moved_to_test >= num_files_to_move_to_test):
                    break
                
                # ensure we do not move the same file twice
                if (filepath in files_moved or filepath in not_dense_files):
                    continue

                trajectory_cell_df = get_trajectory_df_from_txt(filepath)
                
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
                    not_dense_files.append(os.path.basename(filepath))
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
                sys.stdout.write(f"\rMoved {num_files_moved_to_test}/{num_files_to_move_to_test} to test")
                sys.stdout.flush()

            if (num_files_moved_to_test < num_files_to_move_to_test):
                random_files_to_move = random.sample(all_files_with_cells, num_files_to_move_to_test)

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

def move_test_and_validation_back():
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'
    directories_with_moved_files = set()

    cells_df = pd.read_csv(CELL_TXT)
    cells_df['geometry'] = cells_df.apply(lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']), axis=1)
    cells_gdf = gpd.GeoDataFrame(cells_df, geometry='geometry', crs="EPSG:4326")

    validation_files = list(Path(INPUT_VALIDATION_DATA_ORIGINAL_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    test_files = list(Path(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER).rglob('*.txt')) # List all files in the directory recursively
    all_files = [str(path) for path in (validation_files+test_files)] # Convert Path objects to strings
    num_files_to_move = len(all_files)

    brunsbuettel_to_kiel_gdf = gpd.GeoDataFrame([1], geometry=[brunsbuettel_to_kiel_polygon], crs="EPSG:4326").geometry.iloc[0]
    aalborg_harbor_to_kattegat_gdf = gpd.GeoDataFrame([1], geometry=[aalborg_harbor_to_kattegat_bbox], crs="EPSG:4326").geometry.iloc[0]
    doggersbank_to_lemvig_gdf = gpd.GeoDataFrame([1], geometry=[doggersbank_to_lemvig_bbox], crs="EPSG:4326").geometry.iloc[0] 
    skagen_gdf = gpd.GeoDataFrame([1], geometry=[skagens_harbor_bbox], crs="EPSG:4326").geometry.iloc[0]

    areas = [
        (brunsbuettel_to_kiel_gdf, 'brunsbuettel_to_kiel'), 
        (aalborg_harbor_to_kattegat_gdf, 'aalborg_harbor_to_kattegat'), 
        (doggersbank_to_lemvig_gdf, 'doggersbank_to_lemvig'),
        (skagen_gdf, 'skagen_harbor')]

    for i, file_path in enumerate(all_files, start=1):
        vessel_mmsi_folder = f'{file_path.split(os_path_split)[-3]}/{file_path.split(os_path_split)[-2]}'
        cell_trajectory_df = get_trajectory_df_from_txt(file_path)
        area_trajectory_df = cell_trajectory_df.copy(deep=True)

        # For grid cells
        points_cells_harbors = gpd.sjoin(cell_trajectory_df, cells_gdf, how="left", predicate="intersects", lsuffix='left', rsuffix='right')

        for (cell_id, group) in points_cells_harbors.groupby('cell_id'):
            if group.empty:
                continue

            output_folder = INPUT_GRAPH_CELLS_FOLDER
            output_folder = os.path.join(output_folder, str(cell_id))
            output_folder = os.path.join(output_folder, vessel_mmsi_folder)

            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, os.path.basename(file_path))
            group[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 

        # for input area
        for (area, name) in areas:
                area_trajectory_df['within_boundary_box'] = area_trajectory_df.within(area)            
                change_detected = area_trajectory_df['within_boundary_box'] != area_trajectory_df['within_boundary_box'].shift(1)
                area_trajectory_df['group'] = change_detected.cumsum()
                
                # Find the largest group within the boundary box
                group_sizes = area_trajectory_df[area_trajectory_df['within_boundary_box']].groupby('group').size()
                valid_groups = group_sizes[group_sizes >= 2]

                if valid_groups.empty:
                    continue
            
                largest_group_id = valid_groups.idxmax()

                # Filter trajectory points based on the largest group within the boundary box
                trajectory_filtered_df = area_trajectory_df[(area_trajectory_df['group'] == largest_group_id) & area_trajectory_df['within_boundary_box']]
                output_folder = INPUT_GRAPH_AREA_FOLDER
                output_folder = os.path.join(output_folder, name)
                output_folder = os.path.join(output_folder, vessel_mmsi_folder)

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, os.path.basename(file_path))
                trajectory_filtered_df.reset_index(drop=True).to_csv(output_path, sep=',', index=True, header=True, mode='w') 

        # move the back to original folder
        end_dir = os.path.join(INPUT_GRAPH_FOLDER, vessel_mmsi_folder)
        os.makedirs(end_dir, exist_ok=True)
        shutil.move(file_path, end_dir)
        directories_with_moved_files.add(os.path.dirname(file_path))
        sys.stdout.write(f"\rMoved {i}/{num_files_to_move}")
        sys.stdout.flush()
    
    # Remove empty directories
    empty_folders_removed = 0
    for dir_path in directories_with_moved_files:
        if not os.listdir(dir_path):  # Check if directory is empty
            os.rmdir(dir_path)  # Remove empty directory
            empty_folders_removed += 1

    logging.info(f'Finished moving {num_files_to_move} files\n Removed {empty_folders_removed} empty directories from input graph')

print('started moving files back')
move_test_and_validation_back()

print('began creating test and validation data for cells')
move_random_files_to_test_and_validation()
#write_trajectories_for_area(INPUT_VALIDATION_DATA_ORIGINAL_FOLDER, INPUT_VALIDATION_SPARSED_AREA_FOLDER)
#write_trajectories_for_all(INPUT_VALIDATION_DATA_ORIGINAL_FOLDER, INPUT_VALIDATION_SPARSED_ALL_FOLDER)
#write_trajectories_for_area(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER, INPUT_TEST_SPARSED_AREA_FOLDER)
#write_trajectories_for_all(INPUT_TEST_DATA_FOLDER_ORIGINAL_FOLDER, INPUT_TEST_SPARSED_ALL_FOLDER)