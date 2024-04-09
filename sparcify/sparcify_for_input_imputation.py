import os
import sys
import random
import shutil
import time as t
import datetime as dt
from typing import Callable
from .classes import SparsifyResult
from shapely.geometry import box, Polygon
from multiprocessing import freeze_support
from data.logs.logging import setup_logger
from concurrent.futures import ProcessPoolExecutor
from .sparcify_methods import sparcify_realisticly_strict_trajectories, sparcify_trajectories_realisticly, sparcify_large_time_gap_with_threshold_percentage, sparcify_trajectories_randomly_using_threshold

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ORIGINAL_FOLDER = os.path.join(DATA_FOLDER, 'original')
INPUT_GRAPH_FOLDER = os.path.join(DATA_FOLDER, 'input_graph')
INPUT_IMPUTATION_FOLDER = os.path.join(DATA_FOLDER, 'input_imputation')
INPUT_IMPUTATION_ORIGINAL_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'original')
INPUT_ALL_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'all')
INPUT_ALL_VALIDATION_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'validation')
INPUT_ALL_TEST_FOLDER = os.path.join(INPUT_ALL_FOLDER, 'test')
INPUT_AREA_FOLDER = os.path.join(INPUT_IMPUTATION_FOLDER, 'area')
INPUT_AREA_VALIDATION_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'validation')
INPUT_AREA_TEST_FOLDER = os.path.join(INPUT_AREA_FOLDER, 'test')
SPARCIFY_LOG = 'sparcify_log.txt'

logging = setup_logger(name=SPARCIFY_LOG, log_file=SPARCIFY_LOG)

def get_files_in_range(start_date, end_date, directory):
    """
    Lists files in the specified directory with dates in their names falling between start_date and end_date.
    
    :param start_date: Start date in DD-MM-YYYY format
    :param end_date: End date in DD-MM-YYYY format
    :param directory: Directory to search for files
    :return: List of filenames that fall within the date range
    """
    files_in_range = []

    if start_date in '' and end_date in '':
        for path, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(path, filename)
                files_in_range.append(filepath)
        return files_in_range

    start_date = dt.datetime.strptime(start_date, '%d-%m-%Y').date()
    end_date = dt.datetime.strptime(end_date, '%d-%m-%Y').date()

    for path, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Extract the date part from the filename, assuming format like '111219502_01-03-2023_11-45-51'
            parts = filename.split('_')
            if len(parts) < 3:
                logging.error(f'Incorrect nameformat for {filename}')
                quit()  # Not enough parts in the filename
            date_str = parts[1]  # The date part is expected to be in the middle
            try:
                file_date = dt.datetime.strptime(date_str, '%d-%m-%Y').date()
                if start_date <= file_date <= end_date:
                    filepath = os.path.join(path, filename)
                    files_in_range.append(filepath)
            except ValueError:
                # If date parsing fails, ignore the file
                pass

    return files_in_range

def write_trajectories_for_area():
        # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        brunsbuettel_to_rendsburg_bbox:Polygon = box(minx=9.114532, miny=53.880869, maxx=9.722900, maxy=54.314921)
        brunsbuettel_to_rendsburg_path = os.path.join(INPUT_AREA_FOLDER, 'brunsbuettel_to_rendsburg')

        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=brunsbuettel_to_rendsburg_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=brunsbuettel_to_rendsburg_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=brunsbuettel_to_rendsburg_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=brunsbuettel_to_rendsburg_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=brunsbuettel_to_rendsburg_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=brunsbuettel_to_rendsburg_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=brunsbuettel_to_rendsburg_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=brunsbuettel_to_rendsburg_bbox)

        aalborg_harbor_to_kattegat_bbox:Polygon = box(minx=9.841940, miny=56.970433, maxx=10.416415, maxy=57.098774)
        aalborg_harbor_to_kattegat_path = os.path.join(INPUT_AREA_FOLDER, 'aalborg_harbor_to_kattegat')

        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=aalborg_harbor_to_kattegat_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=aalborg_harbor_to_kattegat_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=aalborg_harbor_to_kattegat_bbox)


        doggersbank_to_lemvig_bbox:Polygon = box(minx=3.5, miny=54.5, maxx=8.5, maxy=56.5)
        doggersbank_to_lemvig_path = os.path.join(INPUT_AREA_FOLDER, 'doggersbank_to_lemvig')

        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=doggersbank_to_lemvig_path, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=doggersbank_to_lemvig_path, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=doggersbank_to_lemvig_path, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=doggersbank_to_lemvig_path, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=doggersbank_to_lemvig_bbox)

def write_trajectories_for_all():

    # Wrap the code in if __name__ == '__main__': block and call freeze_support()
    if __name__ == '__main__':
        freeze_support()

        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=INPUT_ALL_TEST_FOLDER, action=sparcify_realisticly_strict_trajectories, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=INPUT_ALL_TEST_FOLDER, action=sparcify_trajectories_realisticly, threshold=0.0, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=INPUT_ALL_TEST_FOLDER, action=sparcify_large_time_gap_with_threshold_percentage, threshold=0.5, boundary_box=None)
        sparcify_trajectories_with_action_for_folder(str_start_date='',str_end_date='',folder_path=INPUT_ALL_TEST_FOLDER, action=sparcify_trajectories_randomly_using_threshold, threshold=0.5, boundary_box=None)


def sparcify_trajectories_with_action_for_folder(
    str_start_date: str,
    str_end_date: str,
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

    file_paths = get_files_in_range(str_start_date, str_end_date, INPUT_IMPUTATION_ORIGINAL_FOLDER)

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


def move_random_files_to_original_imputation(percentage=0.1):
    os_path_split = '/' if '/' in INPUT_GRAPH_FOLDER else '\\'
    all_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(INPUT_GRAPH_FOLDER):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # Calculate the number of files to move
    num_files_to_move = int(len(all_files) * percentage)
    
    # Randomly select the files
    files_to_move = random.sample(all_files, num_files_to_move)
    
    try:
        logging.info('Began moving files')
        # Move the files
        for i, file_path in enumerate(files_to_move, start=1):
            # Move the file to input imputation folder with vessel/mmsi folder structure
            vessel_mmsi_folder = f'{file_path.split(os_path_split)[-3]}/{file_path.split(os_path_split)[-2]}'
            end_dir = os.path.join(INPUT_IMPUTATION_ORIGINAL_FOLDER, vessel_mmsi_folder)
            os.makedirs(end_dir, exist_ok=True)
            shutil.move(file_path, end_dir)
            sys.stdout.write(f"\rMoved {i}/{num_files_to_move}")
            sys.stdout.flush()
        
        logging.info(f'Finished moving {num_files_to_move} files')
    except Exception as e:
        logging.error(f'Error was thrown with {repr(e)}')