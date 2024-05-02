import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from sqlalchemy import create_engine
from data.logs.logging import setup_logger
from db_connection.config import load_config
from utils import calculate_bearing_difference, export_graph_to_geojson, haversine_distance

"""
    Graph input in the form of trajectory points extracted from the raw AIS data
"""
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER, 'input_graph_cells')

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')


LOG_PATH = 'graph_log.txt'
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

        
def extract_original_trajectories(input_folder) -> list:
    try: 
        ais_points = []
        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                gdf_curr:gpd.GeoDataFrame = get_trajectory_df(file_path=file_path)

                if (gdf_curr.empty):
                    continue
                
                if len(gdf_curr) >= 2:
                    points_with_metadata = gdf_curr[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].itertuples(index=False, name=None)
                    ais_points.extend(points_with_metadata)

        return ais_points
    
    except Exception as e:
        logging.warning(f'Error occurred trying to extract trajectories: {repr(e)}')
        return []

def calculate_cog_difference(cog1, cog2):
    """Calculate the absolute difference in COG, adjusted for circular COG values."""
    diff = abs(cog1 - cog2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def geometric_sampling(trajectories, min_distance_threshold):

    print("Performing geometric sampling")
    if trajectories is None or len(trajectories) == 0:
        logging.error('No trajectories data provided to geometric_sampling.')
        return []

    # Create a KDTree from the trajectories
    coordinates = np.array([point[:2] for point in trajectories])
    kdtree = cKDTree(coordinates)
    
    # This list will store the indices of the points that are kept
    sampled_indices = []
    # This set will store indices that are too close to already selected points and should be skipped
    excluded_indices = set()

    for i in range(len(coordinates)):
        if i in excluded_indices:
            continue

        # Initially add the current point to the list of sampled indices
        sampled_indices.append(i)
        indices = kdtree.query_ball_point(coordinates[i], min_distance_threshold)

        # Track if an opposite COG point has been kept
        opposite_cog_point_kept = False

        for j in indices:
            if j != i:
                cog_diff = calculate_cog_difference(trajectories[i][4], trajectories[j][4]) 
                if cog_diff > 160 and cog_diff < 205:  
                    if not opposite_cog_point_kept:  
                        sampled_indices.append(j)
                        opposite_cog_point_kept = True

        excluded_indices.update(indices)

    # Filter the trajectories to only include sampled points
    sampled_trajectories = [trajectories[i] for i in sampled_indices]

    print("Number of points after sampling:", len(sampled_trajectories))
    return sampled_trajectories


def create_nodes(sampled_trajectories):
    print("Creating Nodes")
    """
        Receives the geometrically sampled AIS points, assigns them with a avg_depth
        by intersecting with a grid layer. The function then returns the a graph
        containing the nodes with relevant attributes. 
    """
    ais_points_gdf = gpd.GeoDataFrame(
        sampled_trajectories, 
        columns=['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type'],
        geometry=gpd.points_from_xy([item[1] for item in sampled_trajectories], [item[0] for item in sampled_trajectories]),
        crs='EPSG:4326'
        )

    config = load_config()
    engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    
    query = """
        SELECT a.index, a.geometry AS ais_geometry, b.depth, b.geometry AS depth_geometry
        FROM (SELECT row_number() OVER () AS index, geometry FROM ais_points) AS a
        CROSS JOIN LATERAL (
            SELECT depth, geometry
            FROM depth_points
            ORDER BY a.geometry <-> geometry
            LIMIT 1
        ) AS b;
    """

    # Load AIS points into temporary table for query performance
    ais_points_gdf.to_postgis("ais_points", engine, if_exists="replace", index=False)

    # Execute query and load results
    nearest_depths_df = pd.read_sql(query, engine)

    # Join nearest depths back to original AIS points DataFrame using the index for correct alignment
    ais_points_gdf = ais_points_gdf.merge(nearest_depths_df, left_index=True, right_on='index')

    # Assign the nearest depth directly
    ais_points_gdf['avg_depth'] = ais_points_gdf['depth']

    ais_points_gdf = ais_points_gdf[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'avg_depth', 'geometry']]
    

    G = nx.Graph()
    for index, row in ais_points_gdf.iterrows():
        node = (row['latitude'], row['longitude'])
        attributes = row.drop(['geometry']).to_dict()
        G.add_node(node, **attributes)

    return G


def create_edges(G, edge_radius_threshold, bearing_threshold, nodes_file_path, edges_file_path):
    print("Creating Edges")
    """
        Creates edges between nodes in the graph. The creation of an edge 
        is based on two criterias. Distance between nodes, and bearing between nodes.
    """
    node_coords_list = [(node, data) for node, data in G.nodes(data=True)]
    node_array = np.array([node for node, data in node_coords_list])
    
    kdtree = cKDTree(node_array)
    edge_count = 0

    for i, (node, data) in enumerate(node_coords_list):
        node_cog = data['cog']
        nearby_indices = kdtree.query_ball_point(node, edge_radius_threshold)
        
        for nearby_index in nearby_indices:
            if nearby_index != i: 
                nearby_node, nearby_data = node_coords_list[nearby_index]
                nearby_cog = nearby_data['cog']

                cog_diff = calculate_bearing_difference(node_cog, nearby_cog)
                
                if cog_diff <= bearing_threshold:
                    distance = haversine_distance(node[0], node[1], nearby_node[0], nearby_node[1])
                    G.add_edge(node, nearby_node, weight=distance)
                    edge_count += 1

    print(f"Total edges created: {edge_count} \n")

    export_graph_to_geojson(G, nodes_file_path, edges_file_path)


def create_graphs_for_cells(node_threshold, edge_threshold, cog_threshold):
    for cell_name in os.listdir(INPUT_FOLDER_PATH):
        folder_path = os.path.join(INPUT_FOLDER_PATH, cell_name)

        """
            Where the created graphs will be placed
        """
        output_folder_path = os.path.join(OUTPUT_FOLDER, f'output/{node_threshold}_{edge_threshold}_{cog_threshold}')

        if os.path.isdir(folder_path):
            output_subfolder = os.path.join(output_folder_path, cell_name)

            # Check if the output folder already exists
            if os.path.exists(output_subfolder):
                print(f"Graph already exists for {cell_name}, skipping...")
                continue

        if os.path.isdir(folder_path):
            print(f"Processing {cell_name}")
            trajectories = extract_original_trajectories(folder_path)

            if len(trajectories) == 0:
                    print(f"No trajectories found in {cell_name}, skipping...")
                    continue
            
            print(f"Number of points in folder {cell_name}:", len(trajectories))

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            output_subfolder = os.path.join(output_folder_path, cell_name)

            os.makedirs(output_subfolder)

            nodes_file_path = os.path.join(output_subfolder, 'nodes.geojson')
            edges_file_path = os.path.join(output_subfolder, 'edges.geojson')

            geometric_sampled_nodes = geometric_sampling(trajectories, node_threshold)
            nodes = create_nodes(geometric_sampled_nodes)
            create_edges(nodes, edge_threshold, cog_threshold, nodes_file_path, edges_file_path) 

create_graphs_for_cells(0.001, 0.001, 45)
create_graphs_for_cells(0.001, 0.002, 45)
create_graphs_for_cells(0.001, 0.004, 45)
create_graphs_for_cells(0.001, 0.005, 45)









def load_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_graph_from_geojson(nodes_geojson_path, edges_geojson_path):
    if not os.path.exists(nodes_geojson_path) or not os.path.exists(edges_geojson_path):
        return None

    G = nx.Graph()
    
    # Load GeoJSON files
    nodes_geojson = load_geojson(nodes_geojson_path)
    edges_geojson = load_geojson(edges_geojson_path)
    
    # Add nodes
    for feature in nodes_geojson['features']:
        node_id = tuple(feature['geometry']['coordinates'][::-1])  
        G.add_node(node_id, **feature['properties'])
    
    # Add edges
    for feature in edges_geojson['features']:
        start_node = tuple(feature['geometry']['coordinates'][0][::-1])  
        end_node = tuple(feature['geometry']['coordinates'][1][::-1])  
        G.add_edge(start_node, end_node, **feature['properties'])
    
    return G

def get_neighbors(cell_id, cells_df):
    row, col = map(int, cell_id.split('_'))
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nbr_row = row + dr
            nbr_col = col + dc
            nbr_id = f"{nbr_row}_{nbr_col}"
            if nbr_id in cells_df.index:
                neighbors.append(nbr_id)
    return neighbors

def process_all_cells(cells_df, threshold_distance):
    GRAPH_INPUT_FOLDER = os.path.dirname(os.path.dirname(__file__)) + f'\\graph_construction_module\\output\\{NODE_THRESHOLD}_{EDGE_THRESHOLD}_{COG_THRESHOLD}'
    PROCESSED_CELLS = set() 

    for cell_id in cells_df.index:
        if cell_id not in PROCESSED_CELLS:
            connect_graphs(cell_id, cells_df, GRAPH_INPUT_FOLDER, threshold_distance, PROCESSED_CELLS)

def connect_graphs(base_cell_id, cells_df, GRAPH_INPUT_FOLDER, threshold_distance, PROCESSED_CELLS):

    base_graph = create_graph_from_geojson(
        os.path.join(GRAPH_INPUT_FOLDER, f"{base_cell_id}\\nodes.geojson"),
        os.path.join(GRAPH_INPUT_FOLDER, f"{base_cell_id}\\edges.geojson")
    )
    if not base_graph:
        print(f"Graph {base_cell_id} has no data")
        return
    
    print(f"Processing {base_cell_id}")

    neighbors = get_neighbors(base_cell_id, cells_df)
    PROCESSED_CELLS.add(base_cell_id)

    for neighbor_id in neighbors:
        if neighbor_id not in PROCESSED_CELLS:  
            neighbor_nodes_path = os.path.join(GRAPH_INPUT_FOLDER, f"{neighbor_id}/nodes.geojson")
            neighbor_edges_path = os.path.join(GRAPH_INPUT_FOLDER, f"{neighbor_id}/edges.geojson")
            if os.path.exists(neighbor_nodes_path) and os.path.exists(neighbor_edges_path):
                neighbor_graph = create_graph_from_geojson(neighbor_nodes_path, neighbor_edges_path)
                if neighbor_graph:
                    connect_two_graphs(base_graph, neighbor_graph, base_cell_id, neighbor_id, threshold_distance)

def connect_two_graphs(G1, G2, base_cell_id, neighbor_id, threshold_distance):
    G1_nodes = [(node, data['longitude'], data['latitude']) for node, data in G1.nodes(data=True) if 'longitude' in data and 'latitude' in data]
    G2_nodes = [(node, data['longitude'], data['latitude']) for node, data in G2.nodes(data=True) if 'longitude' in data and 'latitude' in data]
    tree_G2 = cKDTree([(lon, lat) for _, lon, lat in G2_nodes])

    new_edges_G1 = []
    new_edges_G2 = []

    for node1, lon1, lat1 in G1_nodes:
        nearby_indices = tree_G2.query_ball_point([lon1, lat1], threshold_distance)
        for index in nearby_indices:
            node2, lon2, lat2 = G2_nodes[index]
            distance = np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
            if distance <= threshold_distance:
                new_edges_G1.append((node1, node2, distance))
                new_edges_G2.append((node2, node1, distance))

    for node1, node2, dist in new_edges_G1:
        G1.add_edge(node1, node2, weight=dist)

    for node2, node1, dist in new_edges_G2:
        G2.add_edge(node2, node1, weight=dist)

    output_subfolder = os.path.join(OUTPUT_FOLDER_PATH, f'{base_cell_id}')
    output_subfolder1 = os.path.join(OUTPUT_FOLDER_PATH, f'{neighbor_id}')

    nodes_file_path_g1 = os.path.join(output_subfolder, f'nodes.geojson')
    edges_file_path_g1 = os.path.join(output_subfolder, f'edges.geojson')
    nodes_file_path_g2 = os.path.join(output_subfolder1, f'nodes.geojson')
    edges_file_path_g2 = os.path.join(output_subfolder1, f'edges.geojson')
    
    # Update GeoJSON files
    export_graph_to_geojson(G1, nodes_file_path_g1, edges_file_path_g1)
    export_graph_to_geojson(G2, nodes_file_path_g2, edges_file_path_g2)


# IMPUTATION_INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# CELLS = os.path.join(IMPUTATION_INPUT_FOLDER, 'cells.txt')
# cells_data = pd.read_csv(CELLS, index_col='cell_id')
# process_all_cells(cells_data, 0.001)




