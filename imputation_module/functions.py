import json
from math import radians, sin, cos, sqrt, atan2
import networkx as nx

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
    for start_node, end_node, data in G.edges(data=True):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]  # Note: GeoJSON uses [longitude, latitude]
            },
            "properties": {
                "weight": data.get('weight', None)
            }
        }
        edges_features.append(feature)

    edges_geojson = {
        "type": "FeatureCollection",
        "features": edges_features
    }

    with open(edges_file_path, 'w') as f:
        json.dump(edges_geojson, f)

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

def haversine(coord1, coord2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c
    return distance

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
    nodes_within = []
    lat1, lon1 = point  # Unpack the point tuple

    for node in G.nodes:
        node_data = G.nodes[node]
        # Assume node is a tuple (latitude, longitude)
        lat2, lon2 = node  # Unpack node identifier
        # Calculate distance using the haversine function or another appropriate distance function
        distance = haversine((lat1, lon1), (lat2, lon2))
        if distance <= radius:
            nodes_within.append(node)

    return nodes_within