import os
import pandas as pd
from graph_construction_module.graph import create_graphs_for_cells, process_all_cells
from imputation_module.imputation import impute_trajectory
from evaluation.compare import find_all_and_compare

def process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    file_count = 0
    for root, dirs, files in os.walk(sparse_trajectories):
        for file_name in files:
            if file_name.endswith('.txt'):
                if file_count < 20:
                    file_path = os.path.join(root, file_name)
                    print(f"Imputing trajectory {file_name}")
                    impute_trajectory(file_name, file_path, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)
                else:
                    break
        if file_count >= 20:
            break
                # imputed_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI/data/output_imputation/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
                # original_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI/data/input_imputation/test/original')
                # find_all_and_compare(imputed_trajectories, original_trajectories, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, size, type)


CELLS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI//data//cells.txt')

node_dist_threshold = 0.001
edge_dist_threshold = [0.002] # 0.001, 0.002, 0.003, 0.004, 0.005
cog_angle_threshold = 45
graph_output_name = 'skagen'
graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//output_graph//{graph_output_name}_{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
    
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
                
            process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold)
