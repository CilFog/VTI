import json
from math import radians, sin, cos, sqrt, atan2
import networkx as nx

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