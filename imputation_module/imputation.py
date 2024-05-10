import os
import csv
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from typing import List, Tuple
from data.logs.logging import setup_logger
from shapely.geometry import Point, box, Polygon, LineString
from utils import calculate_interpolated_timestamps, haversine_distance, heuristics, adjust_edge_weights_for_draught, adjust_edge_weights_for_cog, nodes_within_radius, nodes_to_geojson, edges_to_geojson

LOG_PATH = 'imputation_log.txt'

GRAPH_INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')
IMPUTATION_MODULE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imputation_module')
OUTPUT_FOLDER = os.path.join(IMPUTATION_MODULE_FOLDER, 'output')
OUTPUT_FOLDER_RAW = os.path.join(OUTPUT_FOLDER, 'raw')
OUTPUT_FOLDER_PROCESSED = os.path.join(OUTPUT_FOLDER, 'processed')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CELLS_TXT = os.path.join(DATA_FOLDER, 'cells.txt')

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

if not os.path.exists(OUTPUT_FOLDER_RAW):
    os.makedirs(OUTPUT_FOLDER_RAW)

if not os.path.exists(OUTPUT_FOLDER_PROCESSED):
    os.makedirs(OUTPUT_FOLDER_PROCESSED)

def load_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def create_graph_from_geojson(nodes_geojson_path, edges_geojson_path):
    try: 
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
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve graph: {repr(e)}')

# Load GeoJSON files and create a graph
def load_geojson_to_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, **feature['properties'])
        for edge in feature.get('edges', []):
            G.add_edge(coords, tuple(edge['node']), **edge['properties'])
    return G

def merge_graphs(G1, G2):
    for node, attrs in G2.nodes(data=True):
        # Check if necessary attributes exist and are not None
        if attrs.get('latitude') is not None and attrs.get('longitude') is not None:
            # Add the node to G1 only if it has valid latitude and longitude
            G1.add_node(node, **attrs)

    # Iterate over all edges in G2
    for u, v, attrs in G2.edges(data=True):
        # Add the edge to G1 only if both nodes exist in G1 (ensuring both have valid lat/lon)
        if G1.has_node(u) and G1.has_node(v):
            G1.add_edge(u, v, **attrs)
    
    return G1

def find_relevant_cells(trajectory_points, cells_df):
    relevant_cell_ids = set()
    for trajectory in trajectory_points:
        point = Point(trajectory['properties']['longitude'], trajectory['properties']['latitude'])
        # Check which cells contain the point
        for index, row in cells_df.iterrows():
            cell_polygon = box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat'])
            if cell_polygon.contains(point):
                relevant_cell_ids.add(row['cell_id'])
    return list(relevant_cell_ids)

# TODO output_folder_path should be corrected to the correct path
def impute_trajectory(file_name, file_path, graphs, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    start_time = time.time()

    G = nx.Graph()
    cells_df = pd.read_csv(CELLS_TXT) 

    trajectory_points = []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                trajectory_point = {
                    "properties": {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "timestamp": float(row["timestamp"]),
                        "sog": float(row["sog"]),
                        "cog": float(row["cog"]),
                        "draught": float(row["draught"]),
                        "ship_type": row["ship_type"],
                    }
                }
                trajectory_points.append(trajectory_point)
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory to impute: {repr(e)}')

    relevant_cell_ids = find_relevant_cells(trajectory_points, cells_df)
    
    for cell_id in relevant_cell_ids:
        node_path = os.path.join(GRAPH_INPUT_FOLDER, f"{graphs}\\{cell_id}\\nodes.geojson")
        edge_path = os.path.join(GRAPH_INPUT_FOLDER, f"{graphs}\\{cell_id}\\edges.geojson")
        G_cell = create_graph_from_geojson(node_path, edge_path)
        G = merge_graphs(G, G_cell)

    G_apply_cog_penalty = None
    
    imputed_paths = []

    for i in range(len(trajectory_points) - 1):
        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])
        
        if i % 50 == 0:
            print(f"Done with {i} out of {len(trajectory_points)}")

        if start_point not in G:
            G.add_node(start_point, **start_props) 
        if end_point not in G:
            G.add_node(end_point, **end_props)

        for node in nodes_within_radius(G, start_point, edge_dist_threshold):
            if node != start_point:  
                distance = haversine_distance(start_point[0], start_point[1], node[0], node[1])
                G.add_edge(start_point, node, weight=distance)
                G.add_edge(node, start_point, weight=distance)

        for node in nodes_within_radius(G, end_point, edge_dist_threshold): 
            if node != end_point: 
                distance = haversine_distance(end_point[0], end_point[1], node[0], node[1])
                G.add_edge(end_point, node, weight=distance)
                G.add_edge(node, end_point, weight=distance)
        
        direct_path_exists = G.has_edge(start_point, end_point)
        
        if direct_path_exists:
            path = [start_point, end_point]
            imputed_paths.append(path)
        
        else:
            max_draught = start_props.get("draught", None)
            G_apply_draught_penalty = adjust_edge_weights_for_draught(G, start_point, end_point, max_draught)
            G_apply_cog_penalty = adjust_edge_weights_for_cog(G_apply_draught_penalty, start_point, end_point)
            
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

            except nx.NetworkXNoPath:
                distance = haversine_distance(start_props["latitude"], start_props["longitude"], end_props["latitude"], end_props["longitude"])
                G_apply_cog_penalty.add_edge(start_point, end_point, weight=distance)
                imputed_paths.append([start_point, end_point]) 
                    

    unique_nodes = []
    seen_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            if node not in seen_nodes:
                unique_nodes.append(node)
                seen_nodes.add(node) 
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))

    output_folder_path = os.path.join(OUTPUT_FOLDER_RAW, f'{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}//raw//{file_name}')

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    imputed_nodes_file_path = os.path.join(output_folder_path, f'{file_name}_nodes.geojson')
    imputed_edges_file_path = os.path.join(output_folder_path, f'{file_name}_edges.geojson')

    nodes_to_geojson(G_apply_cog_penalty, unique_nodes, imputed_nodes_file_path)
    edges_to_geojson(G_apply_cog_penalty, edges, imputed_edges_file_path)

    end_time = time.time()
    execution_time = end_time - start_time  

    stats = {
        'file_name': file_name,
        'trajectory_points': len(trajectory_points),
        'imputed_paths': len(unique_nodes),
        'execution_time_seconds': execution_time
    }

    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data//stats//solution_stats')
    stats_file = os.path.join(output_directory, f'{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}_imputation.csv')

    write_header = not os.path.exists(stats_file)

    with open(stats_file, mode='a', newline='') as csvfile:
        fieldnames = ['file_name', 'trajectory_points', 'imputed_paths', 'execution_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()  
        writer.writerow(stats)

    return imputed_paths

def calculate_center_position(positions:List[Tuple[float,float]]):
    """Calculate the center position of the trajectory segment."""
    
    if len(positions) < 3:
        print('Error: At least three positions are required to calculate the center position.')
    center_pos = (
        (positions[0][0] + positions[2][0]) / 2,
        (positions[0][1] + positions[2][1]) / 2
    )
    return center_pos

def best_fit(segment:List[Tuple[float,float]]):
    """Perform linear least squares regression on a segment of positions."""
    # extract X and Y coordinates for the segment
    y_coords = [position[1] for position in segment]
    x_coords = [position[0] for position in segment]
    matrix = np.vstack([y_coords, np.ones(len(y_coords))]).T
    regression_result = np.linalg.lstsq(matrix, x_coords, rcond=None)

    slope, intercept = regression_result[0]
    residuals = regression_result[1]

    # compute the predicted y-values based on the best-fit line
    fitted_segment = [(y, slope * y + intercept) for y in y_coords]

    return fitted_segment, residuals[0]

def refine_trajectory_double(trajectory: List[Tuple[float,float]], epsilon=1e-7): # includes the last point of previous fit in the next fit
    if len(trajectory) < 3:
        return gpd.GeoDataFrame(geometry=[Point(y, x) for x, y in trajectory])
    
    anchor: int = 0
    window_size: int = 3
    final_trajectory = []
    previous_fit = trajectory[:2]
    turn_detected = False

    while anchor + window_size <= len(trajectory):
        # extract the current segment
        current_segment:list = trajectory[anchor:anchor + window_size]

        # compute best fit for the current segment
        best_fit_segment, residual = best_fit(current_segment)

        if residual > epsilon and anchor + window_size < len(trajectory):
            extended_segment = trajectory[anchor:anchor + window_size - 1]
            extended_segment = np.append(extended_segment, [trajectory[anchor + window_size]], axis=0)

            _, residual = best_fit(extended_segment)

            if residual > epsilon:
                turn_detected = True 
            else:
                new_point = calculate_center_position([extended_segment[-2], current_segment[-1], extended_segment[-1]])
                current_segment[-1] = new_point
                best_fit_segment, residual = best_fit(current_segment)

        if residual > epsilon:
            turn_detected = False
            final_trajectory.extend(previous_fit[:-1])
            anchor += window_size - 1
            window_size = 3
            previous_fit = [previous_fit[-1]] + trajectory[anchor:anchor + 1]
        else:
            previous_fit = best_fit_segment
            window_size += 1

    # Add the last refined sub-trajectory
    final_trajectory.extend(previous_fit)
    # Convert refined points back to a GeoDataFrame
    refined_geometries = [Point(y, x) for x, y in final_trajectory]

    return gpd.GeoDataFrame(geometry=refined_geometries)

def refine_trajectory(trajectory: List[Tuple[float,float]], epsilon=1e-7):
    if len(trajectory) < 3:
        return gpd.GeoDataFrame(geometry=[Point(y, x) for x, y in trajectory])
    
    anchor: int = 0
    window_size: int = 3
    final_trajectory = []
    previous_fit = trajectory[:2]
    turn_detected = False

    while anchor + window_size <= len(trajectory):
        # extract the current segment
        current_segment:list = trajectory[anchor:anchor + window_size]

        # compute best fit for the current segment
        best_fit_segment, residual = best_fit(current_segment)

        if residual > epsilon and anchor + window_size < len(trajectory):
            extended_segment = trajectory[anchor:anchor + window_size - 1]
            extended_segment = np.append(extended_segment, [trajectory[anchor + window_size]], axis=0)

            _, residual = best_fit(extended_segment)

            if residual > epsilon:
                turn_detected = True 
            else:
                new_point = calculate_center_position([extended_segment[-2], current_segment[-1], extended_segment[-1]])
                current_segment[-1] = new_point
                best_fit_segment, residual = best_fit(current_segment)

        if (turn_detected):
            turn_detected = False
            final_trajectory.extend(previous_fit)
            anchor += window_size
            window_size = 3
            previous_fit = trajectory[anchor:anchor + 2]
        else:
            previous_fit = best_fit_segment
            window_size += 1

    # Add the last refined sub-trajectory
    final_trajectory.extend(previous_fit)
    # Convert refined points back to a GeoDataFrame
    refined_geometries = [Point(y, x) for x, y in final_trajectory]

    return gpd.GeoDataFrame(geometry=refined_geometries)

def find_swapping_point(trip, i, j): # GTI
    # print(trip)
    x1 = list(map(lambda x: x[1], trip[i:j]))
    y1 = list(map(lambda x: x[0], trip[i:j]))
    A1 = np.vstack([x1, np.ones(len(x1))]).T
    function = np.linalg.lstsq(A1, y1, rcond=None)
    m1, c1 = function [0]
    residuals = function[1]
    return residuals, x1, y1, m1, c1

def refinement(trip): # GTI
    new_points = []
    breaking_point = 0
    m1_c1s = []
    # old_bearing = calculate_initial_compass_bearing((trip[0][0], trip[0][1]), (trip[1][0], trip[1][1]))
    for i in range(2, len(trip)):
        if breaking_point == i + 1: 
            continue
        residuals, x1, y1, m1, c1 = find_swapping_point(trip, breaking_point, i)

        if i > breaking_point + 2 and residuals[0] > 1e-7:
        # if i > breaking_point + 2 and abs(old_bearing - new_bearing) > 10:
            m1_c1s.append((prev_m1, prev_c1, i))
            new_points += [(y1[0], x1[0])]
            new_points += [(m1 * xx + c1, xx) for xx in x1[1:-1]]
            new_points += [(y1[-1], x1[-1])]
            breaking_point = i
        prev_m1, prev_c1 = m1, c1
    new_points += [(y1[0], x1[0])]
    new_points += [(m1 * xx + c1, xx) for xx in x1[1:-1]]
    new_points += [(y1[-1], x1[-1])]

        # Convert refined points back to a GeoDataFrame
    refined_geometries = [Point(x, y) for x, y in new_points]
    return gpd.GeoDataFrame(geometry=refined_geometries)

# TODO newfilepath should be correct
def process_imputated_trajectory(filepath_nodes:str):
    nodes_gdf = gpd.read_file(filepath_nodes)
    coordinates = np.column_stack((nodes_gdf['geometry'].map(lambda p: p.x), nodes_gdf['geometry'].map(lambda p: p.y)))

    nodes_refined_gdf = refine_trajectory(coordinates)

    os_path_split = '/' if '/' in filepath_nodes else '\\'
    basedir = os.path.dirname(filepath_nodes).split(os_path_split)[-1]
    filename = os.path.basename(filepath_nodes).split('_nodes')[0]

    new_filepath = os.path.join(OUTPUT_FOLDER_PROCESSED, basedir)
    os.makedirs(new_filepath, exist_ok=True)
    new_filepath = os.path.join(new_filepath, f'{filename}_refined.geojson')

    nodes_refined_gdf.to_file(new_filepath, driver='GeoJSON')

process_imputated_trajectory('/Users/ceciliew.fog/Documents/KandidatSpeciale/VTI/imputation_module/output/raw/209525000_15-01-2024_00-05-59.txt/209525000_15-01-2024_00-05-59.txt_nodes.geojson')