import os
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from sqlalchemy import create_engine
from data.logs.logging import setup_logger
from db_connection.config import load_config
from .functions import calculate_bearing, calculate_bearing_difference, export_graph_to_geojson, haversine_distance

LOG_PATH = 'graph_log.txt'
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')
INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER, 'input_graph/Cargo')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

def get_trajectory_df(file_path) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(file_path, header=0)
        if (df.empty):
            logging.warning('No coordinates to extract')

        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
        return df
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')
        
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
                points_with_metadata = gdf_curr[['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'sog']].itertuples(index=False, name=None)
                ais_points.extend(points_with_metadata)

                processed_mmsi += 1

    return ais_points

def geometric_sampling(trajectories, sampling_radius_threshold, number_of_nodes):
    """
        Iterates over a collection of AIS points and assigns a density score to each point
        based on how many neighboring AIS points is within a specified radius threshold.
        When each point has been assigned a density score, we define how many points we want
        in total. It then randomly exclude points based on the density score, the higher the
        score the greater the change of exclusion is.
    """
    print('Performing geometric sampling')
    coordinates = np.array([point[:2] for point in trajectories])

    # Create a cKDTree for efficient nearest neighbor search
    kdtree = cKDTree(coordinates)

    density_values = np.zeros(len(coordinates), dtype=float)

    # Iterate over each point to calculate density
    for i, point in enumerate(coordinates):
        neighbors = kdtree.query_ball_point(point, sampling_radius_threshold)
        density_values[i] = len(neighbors)

    density_values += 1e-9  # Avoid division by zero

    normalized_density_scores = (density_values - np.min(density_values)) / (np.max(density_values) - np.min(density_values))
    sampling_probabilities = 1 - normalized_density_scores
    sampling_probabilities /= sampling_probabilities.sum()

    num_samples = min(number_of_nodes, len(coordinates))  # Adjust based on your dataset size
    sampled_indices = np.random.choice(len(coordinates), size=num_samples, replace=False, p=sampling_probabilities)
    sampled_ais_points = [trajectories[idx] for idx in sampled_indices]

    return sampled_ais_points

def create_nodes(sampled_trajectories, grid_size):
    """
        Receives the geometrically sampled AIS points, assigns them with a avg_depth
        by intersecting with a grid layer. The function then returns the a graph
        containing the nodes with relevant attributes. 
    """
    print("Creating nodes")

    # Convert sampled points into a GeoDataFrame
    ais_points_gdf = gpd.GeoDataFrame(sampled_trajectories, columns=['latitude', 'longitude', 'cog', 'draught', 'ship_type', 'sog'])
    ais_points_gdf['geometry'] = gpd.points_from_xy(ais_points_gdf.longitude, ais_points_gdf.latitude)
    ais_points_gdf.set_crs(epsg=4326, inplace=True)

    # Assign depth values
    config = load_config()
    engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    parent_grid_id = f"SELECT * FROM {grid_size}"
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

def create_edges(G, edge_radius_threshold, bearing_threshold):
    """
        Creates edges between nodes in the graph. The creation of an edge 
        is based on two criterias. Distance between nodes, and bearing between nodes.
    """
    print("Creating edges")

    node_coords_list = [(node, data) for node, data in G.nodes(data=True)]
    node_array = np.array([node for node, data in node_coords_list])
    
    kdtree = cKDTree(node_array)

    for i, (node, data) in enumerate(node_coords_list):
        node_cog = data['cog']
        nearby_indices = kdtree.query_ball_point(node, edge_radius_threshold)
        
        for nearby_index in nearby_indices:
            if nearby_index != i:  # Exclude self
                nearby_node, nearby_data = node_coords_list[nearby_index]

                bearing = calculate_bearing(node, nearby_node)
                bearing_diff = calculate_bearing_difference(node_cog, bearing)

                if bearing_diff <= bearing_threshold:
                    distance = haversine_distance(node[0], node[1], nearby_node[0], nearby_node[1])
                    G.add_edge(node, nearby_node, weight=distance)
                    G.add_edge(nearby_node, node, weight=distance)
                    
        if i % 50000 == 0:
            print(f"Processed {i} nodes for edge creation.")

    nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'nodes.geojson')
    edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'edges.geojson')

    export_graph_to_geojson(G, nodes_file_path, edges_file_path)

    return G

def create_graph():
    """
        
    """
    graph_trajectories = extract_original_trajectories()
    geometric_sampled_nodes = geometric_sampling(graph_trajectories, 0.001, 500000)
    nodes = create_nodes(geometric_sampled_nodes, 'grid_400')
    edges = create_edges(nodes, 0.01, 45) 
    return edges
