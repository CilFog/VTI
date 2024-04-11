from datetime import timedelta, datetime
import json
from math import radians, sin, cos, sqrt, atan2
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

def nodes_to_geojson(G, nodes, file_path):
    features = []    
    for node in nodes:
        node_properties = G.nodes[node]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]]
            },
            "properties": node_properties
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)


def edges_to_geojson(G, edges, file_path):
    features = []
    for start_node, end_node in edges:
        if G.has_edge(start_node, end_node):
            weight = G[start_node][end_node]['weight']
        else:
            weight = 'undefined'  # Use a placeholder if the edge doesn't exist; adjust as needed

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]
            },
            "properties": {
                "weight": weight  # Add the edge's weight to the properties
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance


def nodes_within_radius(G, point, radius):
    # Extract node positions into a NumPy array (assuming G.nodes is a list of tuples or similar)
    node_positions = np.array([(data['latitude'], data['longitude']) for node, data in G.nodes(data=True)])
    
    # Build a KD-tree for efficient spatial queries
    tree = cKDTree(node_positions)

     # Convert the point to the same format (latitude, longitude) and ensure radius is appropriate
    query_point = np.array([point[0], point[1]])  # Assuming point is already a tuple (latitude, longitude)

    # Use query_ball_point to find indices of nodes within the radius
    indices_within_radius = tree.query_ball_point(query_point, radius)

    # Convert indices back to node identifiers
    nodes_within = [list(G.nodes)[i] for i in indices_within_radius]

    return nodes_within

def adjust_edge_weights_for_draught(G, start_point, end_point, max_draught, base_penalty=1000, depth_penalty_factor=1):
    # Create a copy of the graph so the original graph remains unchanged.
    radius = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])

    relevant_nodes_start = set(nodes_within_radius(G, start_point, radius))
    relevant_nodes_end = set(nodes_within_radius(G, end_point, radius))

    # Union of nodes near start and end points to find all relevant nodes.
    relevant_nodes = relevant_nodes_start.union(relevant_nodes_end)
    
    # Iterate only through edges connected to relevant nodes.
    for node in relevant_nodes:
        for neighbor in G.neighbors(node):
            if neighbor in relevant_nodes:  # Ensure both nodes are relevant
                data = G.get_edge_data(node, neighbor)
                
                # Compute the minimum depth of connected nodes.
                u_depth = G.nodes[node].get('avg_depth', float('inf'))
                v_depth = G.nodes[neighbor].get('avg_depth', float('inf'))
                min_depth = min(u_depth, v_depth)
                
                # Apply penalties based on the draught comparison.
                penalty = base_penalty if abs(min_depth) < max_draught else 0
                
                # Adjust the edge weight accordingly.
                initial_weight = data.get('weight')
                G[node][neighbor]['weight'] = initial_weight + penalty

    return G

def cog_penalty(cog1, cog2, threshold=30, large_penalty=1000):
    angle_difference = abs(cog1 - cog2)
    angle_difference = min(angle_difference, 360 - angle_difference)  # Account for circular nature of angles
    
    # Apply a large penalty if the difference exceeds the threshold
    if angle_difference > threshold:
        return large_penalty
    else:
        return 0

def adjust_edge_weights_for_cog(G, start_point, end_point):
    # Create a copy of the graph so the original graph remains unchanged.
    radius = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])
    
    relevant_nodes_start = set(nodes_within_radius(G, start_point, radius))
    relevant_nodes_end = set(nodes_within_radius(G, end_point, radius))
    
    # Union of nodes near start and end points for all relevant nodes.
    relevant_nodes = relevant_nodes_start.union(relevant_nodes_end)
    
    # Iterate through the relevant nodes and their neighbors to adjust weights selectively.
    for node in relevant_nodes:
        for neighbor in G.neighbors(node):
            if neighbor in relevant_nodes:  # Check if the neighbor is also relevant.
                if 'cog' in G.nodes[node] and 'cog' in G.nodes[neighbor]:
                    # Calculate the COG penalty.
                    cog_u = G.nodes[node]['cog']
                    cog_v = G.nodes[neighbor]['cog']
                    penalty = cog_penalty(cog_u, cog_v)
                    
                    # Get the existing edge data to adjust its weight.
                    data = G.get_edge_data(node, neighbor)
                    initial_weight = data.get('weight')
                    G[node][neighbor]['weight'] = initial_weight + penalty

    return G

def knots_to_meters_per_second(knots):
    return knots * 0.514444

def calculate_interpolated_timestamps(nodes_within_path, start_timestamp_unix, end_timestamp_unix):
    start_timestamp = datetime.fromtimestamp(start_timestamp_unix)
    end_timestamp = datetime.fromtimestamp(end_timestamp_unix)
    
    timestamps = [start_timestamp]
    total_path_length_m = sum(haversine_distance(nodes_within_path[i][0], nodes_within_path[i][1],
                                                 nodes_within_path[i+1][0], nodes_within_path[i+1][1])
                              for i in range(len(nodes_within_path) - 1))

    total_travel_time_seconds = (end_timestamp - start_timestamp).total_seconds()
    
    cumulative_distance_m = 0
    for i in range(1, len(nodes_within_path)):
        edge_length_m = haversine_distance(nodes_within_path[i-1][0], nodes_within_path[i-1][1],
                                           nodes_within_path[i][0], nodes_within_path[i][1])
        cumulative_distance_m += edge_length_m
        
        # Proportion of total path completed
        path_proportion = cumulative_distance_m / total_path_length_m
        
        # Calculate new timestamp based on proportion of path completed
        interpolated_time_seconds = path_proportion * total_travel_time_seconds
        interpolated_timestamp = start_timestamp + timedelta(seconds=interpolated_time_seconds)
        
        timestamps.append(interpolated_timestamp)

    # Make sure the last timestamp is exactly the end timestamp
    timestamps[-1] = end_timestamp

    return [timestamp.strftime('%d/%m/%Y %H:%M:%S') for timestamp in timestamps]