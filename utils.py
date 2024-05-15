import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import json
import math
from math import radians, sin, cos, atan2, degrees, sqrt
from datetime import timedelta, datetime
from scipy.spatial import cKDTree

pd.set_option('future.no_silent_downcasting', True)
THETA_ANGLE_PENALTY = 50

"""
    Functions used in the imputation module
"""
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

def heuristics(coord1, coord2):

    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c

    total_cost = distance

    return total_cost

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

def adjust_edge_weights_for_draught(G, start_point, end_point, max_draught, relevant_nodes, base_penalty=1000, depth_penalty_factor=1):
    # radius = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])

    # radiusNew = radius / 100 

    # relevant_nodes_start = set(nodes_within_radius(G, start_point, radiusNew))
    # relevant_nodes_end = set(nodes_within_radius(G, end_point, radiusNew))

    # relevant_nodes = relevant_nodes_start.union(relevant_nodes_end)

    processed_edges = set()
    for node in relevant_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge not in processed_edges and neighbor in relevant_nodes:
                processed_edges.add(edge)
                data = G.get_edge_data(node, neighbor)
                
                # Calculate minimum depth between nodes
                u_depth = G.nodes[node].get('avg_depth', G.nodes[node].get('draught'))
                v_depth = G.nodes[neighbor].get('avg_depth', G.nodes[neighbor].get('draught'))

                difference = abs(u_depth - v_depth)
                threshold = abs(v_depth * 0.2)
                
                if difference < threshold:
                    penalty = base_penalty
                else:
                    penalty = 0
                # Apply the penalty to the edge's weight
                initial_weight = data.get('weight')
                G[node][neighbor]['weight'] = initial_weight + penalty


    subgraph = G

    return subgraph

def cog_penalty(cog1, cog2, threshold=30, large_penalty=1000):
    angle_difference = abs(cog1 - cog2)
    angle_difference = min(angle_difference, 360 - angle_difference)  # Account for circular nature of angles
    
    # Apply a large penalty if the difference exceeds the threshold
    if angle_difference > threshold:
        return large_penalty
    else:
        return 0

def adjust_edge_weights_for_cog(G, start_point, end_point, relevant_nodes):
    # radius = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])
    
    # radiusNew = radius / 100 

    # relevant_nodes_start = set(nodes_within_radius(G, start_point, radiusNew))
    # relevant_nodes_end = set(nodes_within_radius(G, end_point, radiusNew))
    
    # # Union of nodes near start and end points for all relevant nodes.
    # relevant_nodes = relevant_nodes_start.union(relevant_nodes_end)
    
    # Dictionary to cache penalties between COG values
    penalty_cache = {}

    # Adjust edge weights based on COG values
    for node in relevant_nodes:
        cog_u = G.nodes[node].get('cog')
        if cog_u is None:
            continue
        
        for neighbor in G.neighbors(node):
            if neighbor not in relevant_nodes:
                continue
            
            cog_v = G.nodes[neighbor].get('cog')
            if cog_v is None:
                continue

            # Use cached penalty if available
            cog_pair = (cog_u, cog_v) if cog_u <= cog_v else (cog_v, cog_u)
            if cog_pair in penalty_cache:
                penalty = penalty_cache[cog_pair]
            else:
                penalty = cog_penalty(cog_u, cog_v)
                penalty_cache[cog_pair] = penalty

            # Update the edge weight
            initial_weight = G[node][neighbor].get('weight')
            G[node][neighbor]['weight'] = initial_weight + penalty
    return G

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

"""
    Functions used in the graph module
"""
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

def calculate_initial_compass_bearing(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Summary: Calculates the compas bearing between consecutive rows in a dataframe

    Args:
        df_curr (gpd.GeoDataFrame): dataframe containing all first positions
        df_next (gpd.GeoDataFrame): dataframe containing all next positions

    Returns:
        gpd.GeoDataFrame: df_curr with a bearing
    """

    assert len(df_curr) == len(df_next), "DataFrames must have the same length"
    
    # Calculate bearing
    x = np.sin(df_curr['diff_lon']) * np.cos(df_curr['lat_rad'])     
    y = np.cos(df_next['lat_rad']) * np.sin(df_curr['lat_rad']) - (
        np.sin(df_next['lat_rad']) * np.cos(df_curr['lat_rad']) * np.cos(df_curr['diff_lon'])
    )
    
    # Bearing is now between -180 - 180
    bearing_rad = np.arctan2(x, y)     
    
    # Convert to degrees and normalize such that bearing is between 0 and 360
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360
    
    # Handle the last row 
    bearing_deg.fillna(0, inplace=True)

    return bearing_deg

def get_haversine_dist_df_in_meters(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    return: distance in meters
    """
    assert len(df_curr) == len(df_next), "DataFrames must have the same length"
    
    # Haversine formula
    a = np.sin(df_curr['diff_lat']/2)**2 + np.cos(df_curr['lat_rad']) * np.cos(df_next['lat_rad']) * np.sin(df_curr['diff_lon']/2)**2    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # Radius of earth in kilometers is 6371
    distance_km = 6371 * c
    dist = distance_km * 1000
    dist.fillna(0, inplace=True)
    
    return dist

def get_radian_and_radian_diff_columns(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    # Explicitly mark df_curr and df_next as copies to avoid SettingWithCopyWarning
    df_curr = df_curr.copy()
    df_next = df_next.copy()
    
    # Convert to radians
    df_curr['lat_rad'] = np.radians(df_curr['latitude'])
    df_curr['lon_rad'] = np.radians(df_curr['longitude'])
    df_next['lat_rad'] = np.radians(df_next['latitude'])
    df_next['lon_rad'] = np.radians(df_next['longitude'])
    
    # Calculate differences
    df_curr['diff_lon'] = df_curr['lon_rad'] - df_next['lon_rad']
    df_curr['diff_lat'] = df_curr['lat_rad'] - df_next['lat_rad']
    
    
    return (df_curr.fillna(0).infer_objects(), df_next)

# Function to calculate distance and adjust based on COG
def adjusted_distance(x, y):
    # Calculate Haversine distance
    haversine_dist = geodesic((x[0], x[1]), (y[0], y[1])).m
    
    # Example COG adjustment: if COG difference is > 45 degrees, increase distance
    cog_diff = abs(x[2] - y[2])
    cog_diff = min(cog_diff, 360-cog_diff)

    # apply the angle penalty
    haversine_dist = np.sqrt(haversine_dist ** 2 + (THETA_ANGLE_PENALTY * cog_diff /180) ** 2)

    return haversine_dist