import csv
import heapq
import os
import networkx as nx
from graph_construction_module.graph import create_graph
from .functions import haversine_distance, nodes_within_radius, nodes_to_geojson, edges_to_geojson, export_graph_to_geojson
from .heuristics import heuristics 

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imputation_module')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')
INPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output/imputation_input.txt')

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

def adjust_edge_weights_for_draught(G, max_draught, base_penalty=1000, depth_penalty_factor=1):
    # Create a copy of the graph so the original graph remains unchanged.
    G_temp = G.copy()
    
    for u, v, data in G.edges(data=True):
        # Get the minimum avg_depth of the two nodes connected by the edge
        u_depth = G.nodes[u].get('avg_depth', float('inf'))
        v_depth = G.nodes[v].get('avg_depth', float('inf'))
        min_depth = min(u_depth, v_depth)
        
        # Calculate penalty based on draught and avg_depth comparison
        if min_depth < max_draught:
            # Apply a large penalty if avg_depth is below the draught
            penalty = base_penalty
        else:
            # Apply a scaled penalty based on the difference between draught and avg_depth
            depth_difference = min_depth - max_draught
            penalty = 0 #depth_difference * depth_penalty_factor
            
        # Adjust the weight of the edge in the temporary graph
        # Ensure there's an initial weight; use 1 or a default value if weights weren't previously set
        initial_weight = data.get('weight', 1)
        G_temp[u][v]['weight'] = initial_weight + penalty

    return G_temp


def impute_trajectory():
    G = create_graph()

    print("Add Trajectory to impute")

    trajectory_points = []

    # Open the file and read the contents
    with open(INPUT_FOLDER_PATH, 'r') as csvfile:
        # Create a DictReader to read the CSV data
        reader = csv.DictReader(csvfile)
        
        # Iterate over the CSV rows
        for row in reader:
            # Convert the relevant fields into the format used previously
            trajectory_point = {
                "properties": {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "timestamp": float(row["timestamp"]),
                    "cog": float(row["cog"]),
                    "draught": float(row["draught"]),
                    "ship_type": row["ship_type"],
                }
            }
            # Append this point to the list of trajectory points
            trajectory_points.append(trajectory_point)

    print("Impute trajectory")

    imputed_paths = []  # List to store paths between consecutive points
    
    # Iterate through pairs of consecutive points
    for i in range(len(trajectory_points) - 1):
        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        # Using (latitude, longitude) as unique identifiers for nodes
        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])
        
        # Ensure start and end points are nodes in the graph, add them if not
        if start_point not in G:
            G.add_node(start_point, **start_props)  # Adding with properties unpacked
        if end_point not in G:
            G.add_node(end_point, **end_props)

        # Connect start and end points to existing nodes within a given radius
        for node in nodes_within_radius(G, start_point, 0.03):
            if node != start_point:  # Avoid self-connections
                distance = haversine_distance(start_point[0], start_point[1], node[0], node[1])
                G.add_edge(start_point, node, weight=distance)
                G.add_edge(node, start_point, weight=distance)

        for node in nodes_within_radius(G, end_point, 0.03):
            if node != end_point:  # Avoid self-connections
                distance = haversine_distance(end_point[0], end_point[1], node[0], node[1])
                G.add_edge(end_point, node, weight=distance)
                G.add_edge(node, end_point, weight=distance)
        
        max_draught = start_props.get("draught", None)

        # Export the initial state of the graph to GeoJSON
        G_temp = adjust_edge_weights_for_draught(G, max_draught)

        # Attempt to find a path using A* algorithm
        try:
            path = nx.astar_path(G_temp, start_point, end_point, heuristic=heuristics, weight='weight')
            imputed_paths.append(path)
            print(f"Path found between point {i} and {i+1}: {path}")
        except nx.NetworkXNoPath:
            print(f"No path between points {i} and {i+1}.")
            imputed_paths.append([])  # Append an empty path to indicate no path found

    unique_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            unique_nodes.add(node)  # Add each node to a set of unique nodes
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))

    imputed_nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'ii-nodes.geojson')
    imputed_edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'ii-edges.geojson')

    nodes_to_geojson(G_temp, unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(G_temp, edges, imputed_edges_file_path)

    return imputed_paths

impute_trajectory()

# TO-DO (Imputation):
    # Given a trajectory, it should traverse the graph
        # No node in the graph has a timestamp, it should therefore be calculated
            # based on the timestamp of the input trajectory
            # based on the speed of the input trajectory
        # it decides which edge to take based on:
            # the depth value of the trajectory considered for imputation