import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from data.logs.logging import setup_logger
import networkx as nx
from math import radians, sin, cos, sqrt, atan2, degrees
from utils import calculate_initial_compass_bearing, get_haversine_dist_df_in_meters, get_radian_and_radian_diff_columns
from config import load_config
from connect import connect
from sqlalchemy import create_engine

LOG_PATH = 'graph_log.txt'
INPUT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'data/original')
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
        
points = []
speed_between_points = []
points_to_consider = 0
error_proned_points = 0
potential_clusters = []

def extract_original_trajectories() -> list:
    print('Extracting trajectories')

    ais_points = []
    processed_mmsi= 0

    for dirpath, dirnames, filenames in os.walk(INPUT_FOLDER_PATH):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            gdf_curr:gpd.GeoDataFrame = get_trajectory_df(file_path=file_path)

            if (gdf_curr.empty):
                continue
            
            if len(gdf_curr) >= 2:
                
                # Extract required columns and convert to list of tuples
                points_with_metadata = gdf_curr[['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'timestamp']].itertuples(index=False, name=None)
                ais_points.extend(points_with_metadata)

                processed_mmsi += 1

        #     if processed_mmsi == 50:
        #             break
            
        # if processed_mmsi == 50:
        #     break

    return ais_points

        





def perform_geometric_sampling_and_create_nodes():
    ais_points = extract_original_trajectories()
    #print(ais_points)
    print('Performing geometric sampling')
    coordinates = np.array([point[:2] for point in ais_points])

    #Create a cKDTree for efficient nearest neighbor search
    kdtree = cKDTree(coordinates)

    # Define parameters for geometric sampling
    radius = 0.002  # Radius within which points will be sampled

    # Initialize an array to store the density values for each point
    density_values = np.zeros(len(coordinates), dtype=float)

    # Iterate over each point
    for i, point in enumerate(coordinates):
        # Query neighbors within the radius
        neighbors = kdtree.query_ball_point(point, r=radius)
        # Assign density value as the count of neighbors
        density_values[i] = len(neighbors)

    # Add a small value to the density scores to avoid zero values
    density_values += 1e-9

    # Normalize density scores to range between 0 and 1
    normalized_density_scores = (density_values - np.min(density_values)) / (np.max(density_values) - np.min(density_values))

    # Invert density scores to obtain sampling probabilities
    sampling_probabilities = 1 - normalized_density_scores

    # Perform sampling based on probabilities
    num_samples = 100000  # Define the number of samples to be taken
    sampling_probabilities /= sampling_probabilities.sum()  # Ensure probabilities sum to 1
    sampled_indices = np.random.choice(len(coordinates), size=num_samples, replace=False, p=sampling_probabilities)

    # Get the sampled points
    sampled_ais_points = [ais_points[idx] for idx in sampled_indices]




    ######### Assign depth value to each node #########
    config = load_config()

    engine = create_engine('postgresql://'+config['user']+':'+config['password']+'@'+config['host']+':'+config['port']+'/'+config['database']+'')

    ais_points_gdf = gpd.GeoDataFrame(sampled_ais_points, columns=['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'timestamp'])  # Add more column names as needed

    # Create the geometry column from latitude and longitude
    ais_points_gdf['geometry'] = gpd.points_from_xy(ais_points_gdf.longitude, ais_points_gdf.latitude)

    
    ais_points_gdf.set_crs(epsg=4326, inplace=True)

    parent_grid_id = f"SELECT * FROM grid_1600_final"
    polygons_gdf = gpd.read_postgis(parent_grid_id, engine, geom_col='geometry')
    joined = gpd.sjoin(ais_points_gdf, polygons_gdf, how='left', op='within')

    # Merge with depth values
    merged = joined.merge(polygons_gdf[['geometry', 'avg_depth']], left_on='index_right', right_index=True)

    # Assign depth values to AIS points
    ais_points_gdf['avg_depth'] = merged['avg_depth_x']

    columns_to_include = ['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'timestamp', 'avg_depth']

    # Specify the file name
    output_file_name = "nodes.txt"
    
    # Create the full output file path
    output_file_path = os.path.join(META_TRAJECTORIES_PATH, output_file_name)

    # Write the selected columns of the GeoDataFrame to a CSV file
    ais_points_gdf[columns_to_include].to_csv(output_file_path, index=False, sep=',')
    
    return ais_points_gdf  

def calculate_bearing_difference(bearing1, bearing2):
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)

def calculate_bearing(point1, point2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])

    # Calculate the difference in longitudes
    dlon = lon2 - lon1

    # Calculate the bearing using the atan2 function
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(y, x)

    # Convert bearing from radians to degrees
    bearing = degrees(bearing)

    # Normalize bearing to range from 0 to 360 degrees
    bearing = (bearing + 360) % 360

    return bearing

def export_edges_coordinates(edges_coordinates, output_file):
    output_file_path = os.path.join(META_TRAJECTORIES_PATH, output_file)
    with open(output_file_path, 'w') as f:
        # Write header
        f.write('start_latitude,start_longitude,end_latitude,end_longitude\n')
        
        # Write edge coordinates
        for start, end in edges_coordinates:
            f.write(f"{start[0]},{start[1]},{end[0]},{end[1]}\n")

def create_edges():
    bearing_tolerance = 30  # Define tolerance for bearing difference in degrees

    gdf = perform_geometric_sampling_and_create_nodes()

    print("Creating edges")

    # Extract relevant columns as a NumPy array
    attributes_array = gdf[['latitude', 'longitude', 'cog']].values

    # Build a k-d tree for efficient spatial querying (using latitude and longitude)
    node_array = np.array(list(zip(attributes_array[:, 1], attributes_array[:, 0])))
    kdtree = cKDTree(node_array)

    edges_coordinates = []

    # Loop over each node
    for node_index, node_attributes in enumerate(attributes_array):
        node_coords = node_array[node_index]
        node_cog = node_attributes[2]

        # Query the k-d tree to find nearby nodes within the distance threshold
        nearby_node_indices = kdtree.query_ball_point(node_coords, 0.015)  # Example distance threshold

        if node_index % 1000 == 0:
            print(f"Edges created for {node_index} out of 25,000 nodes.")

        for nearby_index in nearby_node_indices:
            if nearby_index != node_index:  # Exclude the current node
                nearby_node_attributes = attributes_array[nearby_index]
                nearby_node_cog = nearby_node_attributes[2]

                # Calculate the bearing between the nodes
                bearing = calculate_bearing((node_attributes[0], node_attributes[1]),
                                            (nearby_node_attributes[0], nearby_node_attributes[1]))

                # Calculate bearing difference between node's COG and the bearing to nearby node
                bearing_diff = calculate_bearing_difference(node_cog, bearing)

                # Check if the nearby node's COG is within the tolerance of the bearing
                if bearing_diff <= bearing_tolerance:
                    edges_coordinates.append((node_coords, node_array[nearby_index]))


    # TO-DO (Graph):
        # Create a node for each point
            # Enhanced with draught value
        # Create edges between nodes based on:
            # Distance between nodes
            # Bearing between nodes within its distance
        

    # Example function to export or further process edges_coordinates
    export_edges_coordinates(edges_coordinates, 'edges.txt')

def impute_trajectory():
    print("Imputation")    

    # TO-DO (Imputation):
        # Given a trajectory, it should traverse the graph
            # No node in the graph has a timestamp, it should therefore be calculated
                # based on the timestamp of the input trajectory
                # based on the speed of the input trajectory
            # it decides which edge to take based on:
                # the depth value of the trajectory considered for imputation

create_edges()


    
    





#ais_points = np.unique(ais_points)
#kd_tree = cKDTree(ais_points)
# Calculate adjusted distance matrix
#distance_matrix = pairwise_distances(data, metric=adjusted_distance)

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

# calculate bearing between consecutive points
#gdf_next = gdf_curr.shift(-1)
#gdf_curr, gdf_next = get_radian_and_radian_diff_columns(gdf_curr, gdf_next)

#bearing_df = calculate_initial_compass_bearing(df_curr=gdf_curr, df_next=gdf_next)
#gdf_curr['cog'] = gdf_curr['cog'].fillna(bearing_df)

#gdf_curr['dist'] = get_haversine_dist_df_in_meters(df_curr=gdf_curr, df_next=gdf_next).fillna(0)
# Calculate the time difference between consecutive points
#time_differences = gdf_next['timestamp'] - gdf_curr['timestamp']

# Calculate speed for points with subsequent points available
#speeds = gdf_curr['dist'] / time_differences
#speeds.fillna(0, inplace=True)

#gdf_curr['speed'] = speeds
                
#gdf_curr[['latitude', 'longitude', 'timestamp', 'cog', 'draught', 'ship_type', 'speed']].to_csv(f'{META_TRAJECTORIES_PATH}/{filename}.txt', sep=',', index=True, header=True, mode='w')     



# remove duplicate points
# centroids:list[Point] = np.unique(points, axis=0)
# kdtree = cKDTree(centroids)

# print('\n\ncomputing nns for all clusters')
# indices_within_radius = kdtree.query_ball_point(x=centroids, r=RADIUS_DEGREE_THRESHOLD, p=2)

