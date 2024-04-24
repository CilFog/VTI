import json
import os
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from tslearn.metrics import dtw

ORGIGNAL_TRAJECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ORGIGNAL_TRAJECTORY_PATH = os.path.join(ORGIGNAL_TRAJECTORY, 'input_imputation\\area\\aalborg_harbor\\large_time_gap_0_5\\Cargo\\209525000_15-01-2024_00-05-59.txt')

IMPUTED_TRAJECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imputation_module')
IMPUTED_TRAJECTORY_PATH = os.path.join(IMPUTED_TRAJECTORY, 'output\\aalborg-nodes.geojson')

def load_geojson_extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        geojson = json.load(f)
    
    coordinates = []
    # Assuming standard GeoJSON format
    for feature in geojson['features']:
        # Assumes coordinates are in [longitude, latitude] format
        lon, lat = feature['geometry']['coordinates']
        coordinates.append([lat, lon])  # Appending as [latitude, longitude]

    return np.array(coordinates)
    
def load_csv_extract_coordinates(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Extract latitude and longitude columns
    coordinates = data[['latitude', 'longitude']]
    
    return coordinates.values

def dynamic_time_warping(original_trajectory, imputed_trajectory):
        
    distance = dtw(original_trajectory, imputed_trajectory)

    print("DTW Distance in Euclidean distance:", distance)

def frechet_distance(original_trajectory, imputed_trajectory):
    
    forward_frechet = directed_hausdorff(original_trajectory, imputed_trajectory)[0]
    reverse_frechet = directed_hausdorff(imputed_trajectory, original_trajectory)[0]

    print("Fréchet Distance in Euclidean:", max(forward_frechet, reverse_frechet))


original_trajectory = load_csv_extract_coordinates(ORGIGNAL_TRAJECTORY_PATH)
imputed_trajectory = load_geojson_extract_coordinates(IMPUTED_TRAJECTORY_PATH)

dynamic_time_warping(original_trajectory, imputed_trajectory)
frechet_distance(original_trajectory, imputed_trajectory)