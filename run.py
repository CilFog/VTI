import os
import pandas as pd
from graph_construction_module.graph import create_graphs_for_cells, process_all_cells
from imputation_module.imputation import load_graphs_and_impute_trajectory, load_intersecting_graphs_and_impute_trajectory
from evaluation.compare import find_all_and_compare
from copy import deepcopy
import networkx as nx
import json

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
        print(f'Error occurred trying to retrieve graph: {repr(e)}')

def load_complete_graph(graph_path):
    G = nx.Graph()
    
    # Walk through the directories in the graph_path
    for root, dirs, files in os.walk(graph_path):
        node_file = None
        edge_file = None
        
        # Check if the current directory has the required geojson files
        for file in files:
            if file == 'nodes.geojson':
                node_file = os.path.join(root, file)
            elif file == 'edges.geojson':
                edge_file = os.path.join(root, file)
        
        # If both files are found, load them into a graph and merge it with the main graph
        if node_file and edge_file:
            G_sub = create_graph_from_geojson(node_file, edge_file)
            G = nx.compose(G, G_sub)  # Merge the current subgraph into the main graph

    return G

def load_all_graph_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    
    original_graph = load_complete_graph(graph_path)
    
    for root, dirs, files in os.walk(sparse_trajectories):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Imputing trajectory {file_name}")
                load_graphs_and_impute_trajectory(file_name, file_path, deepcopy(original_graph), node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)
            
                # imputed_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI/data/output_imputation/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
                # original_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI/data/input_imputation/test/original')
                # find_all_and_compare(imputed_trajectories, original_trajectories, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, size, type)

def load_intersecting_graphs_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    for root, dirs, files in os.walk(sparse_trajectories):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Imputing trajectory {file_name}")
                load_intersecting_graphs_and_impute_trajectory(file_name, file_path, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)
            
                # imputed_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI/data/output_imputation/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
                # original_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI/data/input_imputation/test/original')
                # find_all_and_compare(imputed_trajectories, original_trajectories, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, size, type)



CELLS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI//data//cells.txt')

node_dist_threshold = 0.001
edge_dist_threshold = [0.002] # 0.001, 0.002, 0.003, 0.004, 0.005
cog_angle_threshold = 45
graph_output_name = 'skagen'
    
for edge_dist_threshold in edge_dist_threshold:
    graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//output_graph//{graph_output_name}_{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
    cells_data = pd.read_csv(CELLS, index_col='cell_id')

    """
        Create graphs and connect them
    """
    #create_graphs_for_cells(node_dist_threshold, edge_dist_threshold, cog_angle_threshold, graph_output_name)
    #process_all_cells(cells_data, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, graph_output_name)

    """
        Impute all trajectories in test folder
    """

    sparse = [500, 1000, 2000, 4000, 8000] # 50, 100, 200, 400, 800, 1600, 3200, 6400
    types = ['many_gap', 'single_gap', 'realistic'] #'many_gap', 'single_gap', 'realistic', 'realistic_strict'
    for size in sparse:
        for type in types:
            if types == 'realistic':
                sparse_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//input_imputation//test//sparsed//all//{type}')
            else:
                sparse_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//input_imputation//test//sparsed//all//{type}//{size}')
                
            load_all_graph_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold)
