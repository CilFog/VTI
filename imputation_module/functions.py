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