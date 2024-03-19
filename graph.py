import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from data.logs.logging import setup_logger
from utils import calculate_initial_compass_bearing, get_haversine_dist_df_in_meters, get_radian_and_radian_diff_columns

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
            
            #points.extend(gdf_curr.geometry.tolist())    
            
            if len(gdf_curr) >= 2:
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
                
                #ais_points.extend(gdf_curr[['latitude', 'longitude']].to_numpy())
                # Extract required columns and convert to list of tuples
                points_with_metadata = gdf_curr[['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'timestamp']].itertuples(index=False, name=None)
                ais_points.extend(points_with_metadata)
                # for _, row in gdf_curr.iterrows():
                #     # Append a tuple with coordinates and metadata
                #     ais_points.append((row['latitude'], row['longitude'], row['cog'], row['draught'], row['ship_type'], row['timestamp']))


        #         processed_mmsi += 1

        #     if processed_mmsi == 50:
        #             break
            
        # if processed_mmsi == 50:
        #     break

    return ais_points

        

def perform_geometric_sampling():
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
    num_samples = 25000  # Define the number of samples to be taken
    sampling_probabilities /= sampling_probabilities.sum()  # Ensure probabilities sum to 1
    sampled_indices = np.random.choice(len(coordinates), size=num_samples, replace=False, p=sampling_probabilities)

    # Get the sampled points
    sampled_ais_points = [ais_points[idx] for idx in sampled_indices]

    # Specify the file name
    output_file_name = "sampled_points_small.txt"

    # Create the full output file path
    output_file_path = os.path.join(META_TRAJECTORIES_PATH, output_file_name)

    # Write the sampled points to the text file
    with open(output_file_path, "w") as f:
        for point in sampled_ais_points:
            f.write(f"{point[0]}, {point[1]}, {point[2]}, {point[3]}\n")


perform_geometric_sampling()







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


# remove duplicate points
# centroids:list[Point] = np.unique(points, axis=0)
# kdtree = cKDTree(centroids)

# print('\n\ncomputing nns for all clusters')
# indices_within_radius = kdtree.query_ball_point(x=centroids, r=RADIUS_DEGREE_THRESHOLD, p=2)

