import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from data.logs.logging import setup_logger
from utils import calculate_initial_compass_bearing, get_haversine_dist_df_in_meters, adjusted_distance

LOG_PATH = 'graph_log.txt'
INPUT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'data/input')
META_TRAJECTORIES_PATH = os.path.join(os.path.dirname(__file__), 'data/meta_trajectories')
RADIUS_METER_THRESHOLD = 50
RADIUS_DEGREE_THRESHOLD = RADIUS_METER_THRESHOLD * 10e-6

# Check if the directory exists, if not, create it
if not os.path.exists(META_TRAJECTORIES_PATH):
    os.makedirs(META_TRAJECTORIES_PATH)

logging = setup_logger(LOG_PATH)

def get_trajectory_df(file_path) -> gpd.GeoDataFrame:
    """
    Summary:
        Reads a trajectory txt file and returns is as a dataframe with srid 4326
        
    Args:
        file_path (str): file_path to txt file with trajectory

    Returns:
        gpd.GeoDataFrame: trajectory as a dataframe
    """
    try:
        df = pd.read_csv(file_path, header=0)
        if (df.empty):
            logging.warning('No coordinates to extract')

        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
        return df
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')

def get_radian_and_radian_diff_columns(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    # Convert to radians
    df_curr['lat_rad'] = np.radians(df_curr['latitude'])
    df_curr['lon_rad'] = np.radians(df_curr['longitude'])
    df_next['lat_rad'] = np.radians(df_next['latitude'])
    df_next['lon_rad'] = np.radians(df_next['longitude'])
    
    # Calculate differences
    df_curr['diff_lon'] = df_curr['lon_rad'] - df_next['lon_rad']
    df_curr['diff_lat'] = df_curr['lat_rad'] - df_next['lat_rad']
    
    return (df_curr.fillna(0), df_next)

print('Adding meta data to trajectories')

points = []
speed_between_points = []
points_to_consider = 0
error_proned_points = 0
potential_clusters = []
ais_points = []

for dirpath, dirnames, filenames in os.walk(INPUT_FOLDER_PATH):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        gdf_curr:gpd.GeoDataFrame = get_trajectory_df(file_path=file_path)
        
        if (gdf_curr.empty):
            continue
        
        points.extend(gdf_curr.geometry.tolist())    
        
        if len(gdf_curr) >= 2:
            # calculate bearing between consecutive points
            gdf_next = gdf_curr.shift(-1)
            gdf_curr, gdf_next = get_radian_and_radian_diff_columns(gdf_curr, gdf_next)
            
            bearing_df = calculate_initial_compass_bearing(df_curr=gdf_curr, df_next=gdf_next)
            gdf_curr['cog'] = gdf_curr['cog'].fillna(bearing_df)

            gdf_curr['dist'] = get_haversine_dist_df_in_meters(df_curr=gdf_curr, df_next=gdf_next).fillna(0)
            # Calculate the time difference between consecutive points
            time_differences = gdf_next['timestamp'] - gdf_curr['timestamp']
            
            # # Calculate speed for points with subsequent points available
            speeds = gdf_curr['dist'] / time_differences
            speeds.fillna(0, inplace=True)
            
            gdf_curr['speed'] = speeds
                            
            #gdf_curr[['latitude', 'longitude', 'timestamp', 'cog', 'draught', 'ship_type', 'speed']].to_csv(f'{META_TRAJECTORIES_PATH}/{filename}.txt', sep=',', index=True, header=True, mode='w')     
            
            ais_points.extend(gdf_curr[['latitude', 'longitude']].to_numpy())

# ais_points = np.unique(ais_points)
# kd_tree = cKDTree(ais_points)
# # Calculate adjusted distance matrix
# distance_matrix = pairwise_distances(data, metric=adjusted_distance)

# # Apply DBSCAN
# dbscan = DBSCAN(metric='precomputed', eps=3000, min_samples=2) # Adjust eps based on your needs
# dbscan.fit(distance_matrix)

# # Get cluster labels
# labels = dbscan.labels_

# # Plotting
# plt.figure(figsize=(8, 6))

# # Plotting the points with COG represented by color
# for i in range(data.shape[0]):
#     if labels[i] == -1: # Outlier
#         plt.scatter(data[i, 0], data[i, 1], c='r', marker='o', label='Outlier')
#     else: # Clustered points
#         # Normalize COG values to [0, 1] for color mapping
#         cog_normalized = (data[i, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
#         plt.scatter(data[i, 0], data[i, 1], c=plt.cm.viridis(cog_normalized), marker='o', label='Clustered Point')

# # Adding a legend
# plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10)],
#            labels=['Outlier', 'Clustered Point'])

# plt.title('DBSCAN Clustering with COG')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.show()


# remove duplicate points
# centroids:list[Point] = np.unique(points, axis=0)
# kdtree = cKDTree(centroids)

# print('\n\ncomputing nns for all clusters')
# indices_within_radius = kdtree.query_ball_point(x=centroids, r=RADIUS_DEGREE_THRESHOLD, p=2)

