import os
import networkx as nx
from graph_construction_module.graph import create_graph
from .functions import nodes_within_radius, haversine, nodes_to_geojson, edges_to_geojson
from .heuristics import heuristics 

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imputation_module')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

def impute_trajectory():
    G = create_graph()

    print("Impute trajectories")

    trajectory_points = [
        {"properties": {"latitude": 56.708913, "longitude": 9.174883, "cog": 100, "draught": 5.5, "ship_type": "Passenger", "timestamp": 1709252606.0, "avg_depth": -18.8}},
        {"properties": {"latitude": 56.70362, "longitude": 9.19052, "cog": 105, "draught": 3.2, "ship_type": "Cargo", "timestamp": 1709352606.0, "avg_depth": -15.2}},
        {"properties": {"latitude": 56.70474, "longitude": 9.18772, "cog": 110, "draught": 4.1, "ship_type": "Cargo", "timestamp": 1709452606.0, "avg_depth": -20.0}},
        {"properties": {"latitude": 56.70669, "longitude": 9.18585, "cog": 95, "draught": 5.3, "ship_type": "Passenger", "timestamp": 1709552606.0, "avg_depth": -22.1}},
        {"properties": {"latitude": 56.70812, "longitude": 9.17996, "cog": 90, "draught": 6.0, "ship_type": "Passenger", "timestamp": 1709652606.0, "avg_depth": -18.5}},
        {"properties": {"latitude": 56.70841, "longitude": 9.17677, "cog": 85, "draught": 4.8, "ship_type": "Cargo", "timestamp": 1709752606.0, "avg_depth": -14.3}},
        {"properties": {"latitude": 56.700597, "longitude": 9.191901, "cog": 80, "draught": 5.1, "ship_type": "Cargo", "timestamp": 1709852606.0, "avg_depth": -16.7}}
    ]

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
        for node in nodes_within_radius(G, start_point, 0.06):
            if node != start_point:  # Avoid self-connections
                G.add_edge(start_point, node)
        for node in nodes_within_radius(G, end_point, 0.06):
            if node != end_point:  # Avoid self-connections
                G.add_edge(end_point, node)
        
        # Attempt to find a path using A* algorithm
        try:
            path = nx.astar_path(G, start_point, end_point, heuristic=heuristics)
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


    imputed_nodes_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-nodes.geojson')
    imputed_edges_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'i-edges.geojson')

    nodes_to_geojson(unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(edges, imputed_edges_file_path)

    return imputed_paths

impute_trajectory()

# TO-DO (Imputation):
    # Given a trajectory, it should traverse the graph
        # No node in the graph has a timestamp, it should therefore be calculated
            # based on the timestamp of the input trajectory
            # based on the speed of the input trajectory
        # it decides which edge to take based on:
            # the depth value of the trajectory considered for imputation