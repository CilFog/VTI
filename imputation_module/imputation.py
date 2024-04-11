import csv
import heapq
import json
import os
import networkx as nx
#from graph_construction_module.graph import create_graph
from .functions import calculate_interpolated_timestamps, haversine_distance, adjust_edge_weights_for_draught, adjust_edge_weights_for_cog, nodes_within_radius, nodes_to_geojson, edges_to_geojson
from .heuristics import heuristics 

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imputation_module')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')

IMPUTATION_INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
IMPUTATION_INPUT_FOLDER_PATH = os.path.join(IMPUTATION_INPUT_FOLDER, 'input_imputation/area/passanger-area/large_time_gap_0_5/Passenger/220000054_15-01-2024_05-16-57.txt')

GRAPH_INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')
GRAPH_INPUT_NODES = os.path.join(GRAPH_INPUT_FOLDER, 'output\\graph_all\\100000\\nodes.geojson')
GRAPH_INPUT_EDGES = os.path.join(GRAPH_INPUT_FOLDER, 'output\\graph_all\\100000\\edges.geojson')                       

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

def load_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def create_graph_from_geojson(nodes_geojson_path, edges_geojson_path):
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


def impute_trajectory():
    G = create_graph_from_geojson(GRAPH_INPUT_NODES, GRAPH_INPUT_EDGES)

    print("Add Trajectory to impute")

    trajectory_points = []

    # Open the file and read trajectory to impute
    with open(IMPUTATION_INPUT_FOLDER_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            trajectory_point = {
                "properties": {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "timestamp": float(row["timestamp"]),
                    "cog": float(row["cog"]),
                    "sog": float(row["sog"]),
                    "draught": float(row["draught"]),
                    "ship_type": row["ship_type"],
                }
            }
            trajectory_points.append(trajectory_point)

    print("Impute trajectory")
    
    imputed_paths = []  
    
    for i in range(len(trajectory_points) - 1):
        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])
        
        G_tmep = G

        if start_point not in G_tmep:
            G_tmep.add_node(start_point, **start_props) 
        if end_point not in G_tmep:
            G_tmep.add_node(end_point, **end_props)

        for node in nodes_within_radius(G_tmep, start_point, 0.001):
            if node != start_point:  
                distance = haversine_distance(start_point[0], start_point[1], node[0], node[1])
                G_tmep.add_edge(start_point, node, weight=distance)
                G_tmep.add_edge(node, start_point, weight=distance)

        for node in nodes_within_radius(G_tmep, end_point, 0.001): 
            if node != end_point:  # Avoid self-connections
                distance = haversine_distance(end_point[0], end_point[1], node[0], node[1])
                G_tmep.add_edge(end_point, node, weight=distance)
                G_tmep.add_edge(node, end_point, weight=distance)
        
        direct_path_exists = G_tmep.has_edge(start_point, end_point)

        if direct_path_exists:
            print(f"Direct path exists between {i} and {i+1}. Using direct path.")

            path = [start_point, end_point]
            imputed_paths.append(path)
        
        else:
            max_draught = start_props.get("draught", None)
            G_apply_draught_penalty = adjust_edge_weights_for_draught(G_tmep, start_point, end_point, max_draught)
            G_apply_cog_penalty = adjust_edge_weights_for_cog(G_apply_draught_penalty, start_point, end_point)

            print("Finding path")
            try:
                path = nx.astar_path(G_apply_cog_penalty, start_point, end_point, heuristic=heuristics, weight='weight')

                start_timestamp_unix = start_props["timestamp"]  
                end_timestamp_unix = end_props["timestamp"]  
                
                nodes_within_path = [(start_props["latitude"], start_props["longitude"])] + \
                                    [(G_apply_cog_penalty.nodes[n]["latitude"], G_apply_cog_penalty.nodes[n]["longitude"]) for n in path[1:-1]] + \
                                    [(end_props["latitude"], end_props["longitude"])]
                
                interpolated_timestamps = calculate_interpolated_timestamps(nodes_within_path, start_timestamp_unix, end_timestamp_unix)
                
                for index, node_coordinate in enumerate(nodes_within_path):
                    node = path[index]  
                    if node in G_apply_cog_penalty:
                        node_props = G_apply_cog_penalty.nodes[node]
                        node_props.update({
                            'timestamp': interpolated_timestamps[index],  
                            'sog': start_props["sog"], 
                        })
                    else:
                        print(f"Node {node} not found in graph.")
                
                imputed_paths.append(path)
                print(f"Path found between point {i} and {i+1}: {path}")

            except nx.NetworkXNoPath:
                distance = haversine_distance(start_props["latitude"], start_props["longitude"], end_props["latitude"], end_props["longitude"])
                G_apply_cog_penalty.add_edge(start_point, end_point, weight=distance)
                imputed_paths.append([start_point, end_point]) 
                print(f"Path (or direct edge) handled between point {i} and {i+1}.")

    unique_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            unique_nodes.add(node)  
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))

    imputed_nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-nodes.geojson')
    imputed_edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-edges.geojson')

    nodes_to_geojson(G_apply_cog_penalty, unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(G_apply_cog_penalty, edges, imputed_edges_file_path)

    return imputed_paths

impute_trajectory()