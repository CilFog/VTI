import json
from math import radians, sin, cos, atan2, degrees

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
    for start_node, end_node in G.edges:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]  # Note: GeoJSON uses [longitude, latitude]
            },
            "properties": {}
        }
        edges_features.append(feature)

    edges_geojson = {
        "type": "FeatureCollection",
        "features": edges_features
    }

    with open(edges_file_path, 'w') as f:
        json.dump(edges_geojson, f)
