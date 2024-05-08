import os

import pandas as pd
from graph_construction_module.graph import create_graphs_for_cells, process_all_cells
from imputation_module.imputation import impute_trajectory
from evaluation.compare import find_all_and_compare

CELLS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\data\\cells.txt')

def run():

    """
        Create graphs
    """
    cells_data = pd.read_csv(CELLS, index_col='cell_id')

    create_graphs_for_cells(node_dist_threshold, edge_dist_threshold, cog_angle_threshold)
    process_all_cells(cells_data, node_dist_threshold, edge_dist_threshold, cog_angle_threshold)

    """
        Perform imputation
    """
    graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\graph_construction_module\\output\\skagen_0.001_0.002_45')
    
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\data\\input_imputation\\area\\aalborg_harbor\\random_0_5')):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                impute_trajectory(file_name, file_path, graph_path)

    """
        Evaluate imputation
    """
    original_trajectory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\data\\input_imputation\\area\\aalborg_harbor\\random_0_5')
    imputed_trajectory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\imputation_module\\output')
    find_all_and_compare(imputed_trajectory_path, original_trajectory_path)

node_dist_threshold = 0.001
edge_dist_threshold = 0.001
cog_angle_threshold = 45
graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\graph_construction_module\\output\\skagen_0.001_0.002_45')
original_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\data\\input_imputation\\area\\aalborg_harbor\\random_0_5')
imputed_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI\\imputation_module\\output')


run(node_dist_threshold, edge_dist_threshold, cog_angle_threshold, graph_path, original_trajectories, imputed_trajectories)
