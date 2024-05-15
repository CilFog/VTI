import csv
import json
import os
import networkx as nx
from data.logs.logging import setup_logger
from utils import calculate_interpolated_timestamps, haversine_distance, heuristics, adjust_edge_weights_for_draught, adjust_edge_weights_for_cog, nodes_within_radius, nodes_to_geojson, edges_to_geojson
import time
from shapely.geometry import Point, box
import pandas as pd

LOG_PATH = 'imputation_log.txt'

IMPUTATION_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data//output_imputation')

CELLS = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), 'data//cells.txt')

GRAPH_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

if not os.path.exists(IMPUTATION_OUTPUT):
    os.makedirs(IMPUTATION_OUTPUT)

def load_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def create_graph_from_geojson(nodes_geojson_path, edges_geojson_path):
    try: 
        G = nx.DiGraph()
        
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
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve graph: {repr(e)}')

# Load GeoJSON files and create a graph
def load_geojson_to_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, **feature['properties'])
        for edge in feature.get('edges', []):
            G.add_edge(coords, tuple(edge['node']), **edge['properties'])
    return G

def merge_graphs(G1, G2):
    for node, attrs in G2.nodes(data=True):
        # Check if necessary attributes exist and are not None
        if attrs.get('latitude') is not None and attrs.get('longitude') is not None:
            # Add the node to G1 only if it has valid latitude and longitude
            G1.add_node(node, **attrs)

    # Iterate over all edges in G2
    for u, v, attrs in G2.edges(data=True):
        # Add the edge to G1 only if both nodes exist in G1 (ensuring both have valid lat/lon)
        if G1.has_node(u) and G1.has_node(v):
            G1.add_edge(u, v, **attrs)
    
    return G1

def find_relevant_cells(trajectory_points, cells_df):
    relevant_cell_ids = set()
    for trajectory in trajectory_points:
        point = Point(trajectory['properties']['longitude'], trajectory['properties']['latitude'])
        # Check which cells contain the point
        for index, row in cells_df.iterrows():
            cell_polygon = box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat'])
            if cell_polygon.contains(point):
                relevant_cell_ids.add(row['cell_id'])
    return list(relevant_cell_ids)

def impute_trajectory(file_name, file_path, graphs, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size):
    start_time = time.time()

    G = nx.DiGraph()
    cells_df = pd.read_csv(CELLS) 

    trajectory_points = []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                trajectory_point = {
                    "properties": {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "timestamp": float(row["timestamp"]),
                        "sog": float(row["sog"]),
                        "cog": float(row["cog"]),
                        "draught": float(row["draught"]),
                        "ship_type": row["ship_type"],
                    }
                }
                trajectory_points.append(trajectory_point)
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory to impute: {repr(e)}')

    relevant_cell_ids = find_relevant_cells(trajectory_points, cells_df)
    
    for cell_id in relevant_cell_ids:
        node_path = os.path.join(GRAPH_OUTPUT, f"{graphs}//{cell_id}//nodes.geojson")
        edge_path = os.path.join(GRAPH_OUTPUT, f"{graphs}//{cell_id}//edges.geojson")
        G_cell = create_graph_from_geojson(node_path, edge_path)
        G = merge_graphs(G, G_cell)

    G_apply_cog_penalty = None
    
    imputed_paths = []

    for i in range(len(trajectory_points) - 1):
        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])
        
        if i % 50 == 0:
            print(f"Done with {i} out of {len(trajectory_points)}")
        
        if start_point not in G:
            G.add_node(start_point, **start_props) 
        if end_point not in G:
            G.add_node(end_point, **end_props)

        for node in nodes_within_radius(G, start_point, edge_dist_threshold):
            if node != start_point:  
                distance = haversine_distance(start_point[0], start_point[1], node[0], node[1])
                G.add_edge(start_point, node, weight=distance)
                G.add_edge(node, start_point, weight=distance)

        for node in nodes_within_radius(G, end_point, edge_dist_threshold): 
            if node != end_point: 
                distance = haversine_distance(end_point[0], end_point[1], node[0], node[1])
                G.add_edge(end_point, node, weight=distance)
                G.add_edge(node, end_point, weight=distance)
        
        direct_path_exists = G.has_edge(start_point, end_point)
        
        if direct_path_exists:
            path = [start_point, end_point]
            imputed_paths.append(path)
        
        else:
            max_draught = start_props.get("draught", None)
            #G_apply_draught_penalty = adjust_edge_weights_for_draught(G, start_point, end_point, max_draught)
            G_apply_cog_penalty = G #adjust_edge_weights_for_cog(G_apply_draught_penalty, start_point, end_point)
            
            try:
                path = nx.astar_path(G_apply_cog_penalty, start_point, end_point, heuristic=heuristics, weight='weight')
                imputed_paths.append(path)

            except nx.NetworkXNoPath:
                distance = haversine_distance(start_props["latitude"], start_props["longitude"], end_props["latitude"], end_props["longitude"])
                G_apply_cog_penalty.add_edge(start_point, end_point, weight=distance)
                imputed_paths.append([start_point, end_point]) 
                    
    print("Imputation done")
    unique_nodes = []
    seen_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            if node not in seen_nodes:
                unique_nodes.append(node)
                seen_nodes.add(node) 
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))

    IMPUTATION_OUTPUT_path = os.path.join(IMPUTATION_OUTPUT, f'{type}//{size}//{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}//{file_name}')

    if not os.path.exists(IMPUTATION_OUTPUT_path):
        os.makedirs(IMPUTATION_OUTPUT_path)

    imputed_nodes_file_path = os.path.join(IMPUTATION_OUTPUT_path, f'{file_name}_nodes.geojson')
    imputed_edges_file_path = os.path.join(IMPUTATION_OUTPUT_path, f'{file_name}_edges.geojson')

    nodes_to_geojson(G_apply_cog_penalty, unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(G_apply_cog_penalty, edges, imputed_edges_file_path)

    end_time = time.time()
    execution_time = end_time - start_time  

    stats = {
        'file_name': file_name,
        'trajectory_points': len(trajectory_points),
        'imputed_paths': len(unique_nodes),
        'execution_time_seconds': execution_time
    }

    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data//stats//imputation_stats//{type}//{size}')
    os.makedirs(output_directory, exist_ok=True)
    stats_file = os.path.join(output_directory, f'{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}_imputation.csv')

    write_header = not os.path.exists(stats_file)

    with open(stats_file, mode='a', newline='') as csvfile:
        fieldnames = ['file_name', 'trajectory_points', 'imputed_paths', 'execution_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()  
        writer.writerow(stats)

    return imputed_paths