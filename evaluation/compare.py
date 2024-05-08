import json
import os
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from tslearn.metrics import dtw
import csv

def load_geojson_extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        geojson = json.load(f)
    
    coordinates = []
    for feature in geojson['features']:
        lon, lat = feature['geometry']['coordinates']
        coordinates.append([lat, lon])  

    return np.array(coordinates)
    
def load_csv_extract_coordinates(file_path):
    data = pd.read_csv(file_path)
    coordinates = data[['latitude', 'longitude']]
    
    return coordinates.values

def dynamic_time_warping(original_trajectory, imputed_trajectory):
        
    distance = dtw(original_trajectory, imputed_trajectory)

    return distance

def frechet_distance(original_trajectory, imputed_trajectory):
    
    forward_frechet = directed_hausdorff(original_trajectory, imputed_trajectory)[0]
    reverse_frechet = directed_hausdorff(imputed_trajectory, original_trajectory)[0]

    return max(forward_frechet, reverse_frechet)


def find_all_and_compare(imputed_trajectory_path, original_trajectory_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data//stats//solution_stats')
    stats_file = os.path.join(output_directory, f'{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}_evaluation.csv')
    
    with open(stats_file, mode='w', newline='') as csvfile:
        fieldnames = ['Trajectory', 'Original Length', 'Imputed Length', 'DTW', 'Frechet Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for root, _, files in os.walk(imputed_trajectory_path):
            for file in files:
                if file.endswith('nodes.geojson'):
                    file_path = os.path.join(root, file)
                    base_name = file.replace('_nodes.geojson', '')
                    imputed_trajectory = load_geojson_extract_coordinates(file_path)

                    for r, _, f in os.walk(original_trajectory_path):
                        for ff in f:
                            if ff == base_name:
                                ofile_path = os.path.join(r, ff)
                                original_trajectory = load_csv_extract_coordinates(ofile_path)
                    
                    original_len = len(original_trajectory)
                    imputed_len = len(imputed_trajectory)
                    dtw = dynamic_time_warping(original_trajectory, imputed_trajectory)
                    fd = frechet_distance(original_trajectory, imputed_trajectory)
                    
                    writer.writerow({
                        'Trajectory': base_name,
                        'Original Length': original_len,
                        'Imputed Length': imputed_len,
                        'DTW': dtw,
                        'Frechet Distance': fd
                    })