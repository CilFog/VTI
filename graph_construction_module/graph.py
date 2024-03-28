import json
import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from sqlalchemy import create_engine
from data.logs.logging import setup_logger
from math import radians, sin, cos, sqrt, atan2, degrees
from db_connection.config import load_config

LOG_PATH = 'graph_log.txt'
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')
INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER, 'original/Passenger')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')

# Check if the directory exists, if not, create it
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

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

    return ais_points

def perform_geometric_sampling_and_create_nodes():
    ais_points = extract_original_trajectories()
    """
    Performs geometric sampling on AIS points and creates graph nodes with depth values.
    """
    print('Performing geometric sampling')
    coordinates = np.array([point[:2] for point in ais_points])

    # Create a cKDTree for efficient nearest neighbor search
    kdtree = cKDTree(coordinates)

    # Define parameters for geometric sampling
    radius = 0.002  # Radius within which points will be sampled

    density_values = np.zeros(len(coordinates), dtype=float)

    # Iterate over each point to calculate density
    for i, point in enumerate(coordinates):
        neighbors = kdtree.query_ball_point(point, r=radius)
        density_values[i] = len(neighbors)

    density_values += 1e-9  # Avoid division by zero

    normalized_density_scores = (density_values - np.min(density_values)) / (np.max(density_values) - np.min(density_values))
    sampling_probabilities = 1 - normalized_density_scores
    sampling_probabilities /= sampling_probabilities.sum()

    num_samples = min(50000, len(coordinates))  # Adjust based on your dataset size
    sampled_indices = np.random.choice(len(coordinates), size=num_samples, replace=False, p=sampling_probabilities)
    sampled_ais_points = [ais_points[idx] for idx in sampled_indices]

    # Convert sampled points into a GeoDataFrame
    ais_points_gdf = gpd.GeoDataFrame(sampled_ais_points, columns=['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'timestamp'])
    ais_points_gdf['geometry'] = gpd.points_from_xy(ais_points_gdf.longitude, ais_points_gdf.latitude)
    ais_points_gdf.set_crs(epsg=4326, inplace=True)

    # Assign depth values
    config = load_config()
    engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    parent_grid_id = "SELECT * FROM grid_1600"
    polygons_gdf = gpd.read_postgis(parent_grid_id, engine, geom_col='geometry')
    joined = gpd.sjoin(ais_points_gdf, polygons_gdf, how='left', op='within')
    ais_points_gdf['avg_depth'] = joined['avg_depth']

    # Create a graph and add nodes with attributes
    G = nx.Graph()
    for index, row in ais_points_gdf.iterrows():
        node = (row['latitude'], row['longitude'])
        attributes = row.drop(['geometry']).to_dict()
        G.add_node(node, **attributes)

    return G

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
    output_file_path = os.path.join(OUTPUT_FOLDER_PATH, output_file)
    with open(output_file_path, 'w') as f:
        # Write header
        f.write('start_latitude,start_longitude,end_latitude,end_longitude\n')
        
        # Write edge coordinates
        for start, end in edges_coordinates:
            f.write(f"{start[0]},{start[1]},{end[0]},{end[1]}\n")

def create_edges():
    bearing_tolerance = 30  # Define tolerance for bearing difference in degrees

    G = perform_geometric_sampling_and_create_nodes()

    node_coords_list = [(node, data) for node, data in G.nodes(data=True)]
    node_array = np.array([node for node, data in node_coords_list])
    attributes_array = np.array([data for node, data in node_coords_list])
    
    kdtree = cKDTree(node_array)

    for i, (node, data) in enumerate(node_coords_list):
        node_cog = data['cog']
        nearby_indices = kdtree.query_ball_point(node, 0.001)
        
        for nearby_index in nearby_indices:
            if nearby_index != i:  # Exclude self
                nearby_node, nearby_data = node_coords_list[nearby_index]
                nearby_cog = nearby_data['cog']

                bearing = calculate_bearing(node, nearby_node)
                bearing_diff = calculate_bearing_difference(node_cog, bearing)

                if bearing_diff <= bearing_tolerance:
                    G.add_edge(node, nearby_node)
                    
        if i % 10000 == 0:
            print(f"Processed {i} nodes for edge creation.")

    return G

def haversine(coord1, coord2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance


def nodes_within_radius(G, point, radius):
    """Find all nodes within a specified radius of a point."""
    nodes_within = [node for node in G.nodes if haversine(point, (float(node[0]), float(node[1]))) <= radius]
    return nodes_within

def export_graph_to_geojson(G, nodes_file_path, edges_file_path):
    # Nodes
    nodes_features = []
    for node in G.nodes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]]  # Note: GeoJSON uses [longitude, latitude]
            },
            "properties": G.nodes[node]
        }
        nodes_features.append(feature)

    nodes_geojson = {
        "type": "FeatureCollection",
        "features": nodes_features
    }

    with open(nodes_file_path, 'w') as f:
        json.dump(nodes_geojson, f)

    # Edges
    edges_features = []
    for start_node, end_node in G.edges:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]  # Note: GeoJSON uses [longitude, latitude]
            },
            "properties": {}
        }
        edges_features.append(feature)

    edges_geojson = {
        "type": "FeatureCollection",
        "features": edges_features
    }

    with open(edges_file_path, 'w') as f:
        json.dump(edges_geojson, f)

def nodes_to_geojson(nodes, file_path):
    features = []
    for node in nodes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]]
            },
            "properties": {}
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

def edges_to_geojson(edges, file_path):
    features = []
    for edge in edges:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[edge[0][1], edge[0][0]], [edge[1][1], edge[1][0]]]
            },
            "properties": {}
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)


def impute_trajectory():

    G = create_edges()

    trajectory_points = [(56.708913, 9.174883),
                         (56.70362, 9.19052),
                         (56.70474, 9.18772),
                         (56.70669, 9.18585),
                         (56.70812, 9.17996),
                         (56.70841, 9.17677),
                         (56.700597, 9.191901)]

    imputed_paths = []  # List to store paths between consecutive points
    
    # Iterate through pairs of consecutive points
    for i in range(len(trajectory_points) - 1):
        start_point = trajectory_points[i]
        end_point = trajectory_points[i + 1]
        
        # Ensure start and end points are nodes in the graph
        if start_point not in G:
            G.add_node(start_point)
        if end_point not in G:
            G.add_node(end_point)

        # Connect start and end points to existing nodes within a given radius
        for node in nodes_within_radius(G, start_point, 0.06):
            if node != start_point:  # Avoid self-connections
                G.add_edge(start_point, node)
        for node in nodes_within_radius(G, end_point, 0.06):
            if node != end_point:  # Avoid self-connections
                G.add_edge(end_point, node)
        
        # Attempt to find a path using A* algorithm
        try:
            path = nx.astar_path(G, start_point, end_point, heuristic=haversine)
            imputed_paths.append(path)
            print(f"Path found between point {i} and {i+1}: {path}")
        except nx.NetworkXNoPath:
            print(f"No path between points {i} and {i+1}.")
            imputed_paths.append([])  # Append an empty path to indicate no path found
    
    # Optionally, export the graph to GeoJSON for visualization
    nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'nodes.geojson')
    edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'edges.geojson')

    export_graph_to_geojson(G, nodes_file_path, edges_file_path)

    unique_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            unique_nodes.add(node)  # Add each node to a set of unique nodes
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))


    imputed_nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-nodes.geojson')
    imputed_edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-edges.geojson')

    # Using the functions to create GeoJSON files
    nodes_to_geojson(unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(edges, imputed_edges_file_path)

    return imputed_paths

    # TO-DO (Imputation):
        # Given a trajectory, it should traverse the graph
            # No node in the graph has a timestamp, it should therefore be calculated
                # based on the timestamp of the input trajectory
                # based on the speed of the input trajectory
            # it decides which edge to take based on:
                # the depth value of the trajectory considered for imputation

impute_trajectory()