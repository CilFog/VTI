import json
from math import radians, sin, cos, atan2, degrees, sqrt

def calculate_bearing_difference(bearing1, bearing2):
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)

def calculate_bearing(point1, point2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])

    # Calculate the difference in longitudes
    dlon = lon2 - lon1

    # Calculate the bearing using the atan2 function
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(y, x)

    # Convert bearing from radians to degrees
    bearing = degrees(bearing)

    # Normalize bearing to range from 0 to 360 degrees
    bearing = (bearing + 360) % 360

    return bearing

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

def export_graph_to_geojson(G, nodes_file_path, edges_file_path):
    # Nodes
    nodes_features = []
    for node in G.nodes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]] 
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
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]
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
