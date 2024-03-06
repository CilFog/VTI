import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from data.logs.logging import setup_logger
from bearing import calculate_initial_compass_bearing, get_haversine_dist_in_meters

LOG_PATH = 'graph_log.txt'
INPUT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'data/input')
META_TRAJECTORIES_PATH = os.path.join(os.path.dirname(__file__), 'data/meta_trajectories')
RADIUS_METER_THRESHOLD = 50
RADIUS_DEGREE_THRESHOLD = RADIUS_METER_THRESHOLD * 10e-6
THETA_ANGLE_PENALTY = 50

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
    
    return (df_curr, df_next)

print('Adding meta data to trajectories')

points = []
speed_between_points = []
points_to_consider = 0
error_proned_points = 0
potential_clusters = []
trajectories = []

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
            gdf_curr['dist'] = get_haversine_dist_in_meters(df_curr=gdf_curr, df_next=gdf_next)

            # Calculate the time difference between consecutive points
            time_differences = gdf_next['timestamp'] - gdf_curr['timestamp']
            
            # Calculate speed for points with subsequent points available
            speeds = gdf_curr['dist'] / time_differences
            speeds.fillna(0, inplace=True)

            gdf_curr['speed'] = speeds
                            
            # gdf_curr[['latitude', 'longitude', 'timestamp', 'cog', 'draught', 'ship_type', 'speed']].to_csv(f'{META_TRAJECTORIES_PATH}/{filename}.txt', sep=',', index=True, header=True, mode='w')     
            trajectories.append(gdf_curr[['latitude', 'longitude', 'cog']])

trajectories_df = pd.concat(trajectories) 
gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(trajectories_df.longitude, trajectories_df.latitude), columns=trajectories_df.columns)



# remove duplicate points
# centroids:list[Point] = np.unique(points, axis=0)
# kdtree = cKDTree(centroids)

# print('\n\ncomputing nns for all clusters')
# indices_within_radius = kdtree.query_ball_point(x=centroids, r=RADIUS_DEGREE_THRESHOLD, p=2)

