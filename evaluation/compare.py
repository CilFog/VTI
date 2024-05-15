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
    
    P_len = len(original_trajectory)
    Q_len = len(imputed_trajectory)
    ca = np.full((P_len, Q_len), -1.0)

    # Distance matrix calculation
    dist_matrix = np.linalg.norm(original_trajectory[:, np.newaxis] - imputed_trajectory[np.newaxis, :], axis=2)

    # Initialize the matrix with infinite values
    dp = np.full((P_len, Q_len), np.inf)
    dp[0, 0] = dist_matrix[0, 0]

    for i in range(P_len):
        for j in range(Q_len):
            if i or j:
                min_val = min(dp[i - 1, j] if i > 0 else np.inf,
                              dp[i, j - 1] if j > 0 else np.inf,
                              dp[i - 1, j - 1] if i > 0 and j > 0 else np.inf)
                dp[i, j] = max(min_val, dist_matrix[i, j])

    return dp[-1, -1]


def find_all_and_compare(imputed_trajectory_path, original_trajectory_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, path, type):
    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data//stats//evaluation//{type}//{path}')
    os.makedirs(output_directory, exist_ok=True)
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