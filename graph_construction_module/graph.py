import os
from shapely.geometry import Point, box
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from sqlalchemy import create_engine
from data.logs.logging import setup_logger
from db_connection.config import load_config
from .functions import calculate_bearing_difference, export_graph_to_geojson, haversine_distance

LOG_PATH = 'graph_log.txt'
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER, 'input_graph_cells/4_10')

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_construction_module')
OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'output')

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

def get_trajectory_df(file_path) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(file_path, header=0)
        if (df.empty):
            logging.warning('No coordinates to extract')

        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
        return df
    
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory csv: {repr(e)}')

        
def extract_original_trajectories(input_folder) -> list:
    try: 
        ais_points = []
        for dirpath, dirnames, filenames in os.walk(input_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                gdf_curr:gpd.GeoDataFrame = get_trajectory_df(file_path=file_path)

                if (gdf_curr.empty):
                    continue
                
                if len(gdf_curr) >= 2:
                    points_with_metadata = gdf_curr[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type']].itertuples(index=False, name=None)
                    ais_points.extend(points_with_metadata)

        print("Number of points:", len(ais_points))

        return ais_points
    except Exception as e:
        logging.warning(f'Error occurred trying to extract trajectories: {repr(e)}')
        return []

def calculate_cog_difference(cog1, cog2):
    """Calculate the absolute difference in COG, adjusted for circular COG values."""
    diff = abs(cog1 - cog2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def geometric_sampling(trajectories, min_distance_threshold):

    print("Performing geometric sampling")
    if trajectories is None or len(trajectories) == 0:
        logging.error('No trajectories data provided to geometric_sampling.')
        return []

    # Create a KDTree from the trajectories
    coordinates = np.array([point[:2] for point in trajectories])
    kdtree = cKDTree(coordinates)
    
    # This list will store the indices of the points that are kept
    sampled_indices = []
    # This set will store indices that are too close to already selected points and should be skipped
    excluded_indices = set()

    for i in range(len(coordinates)):
        if i in excluded_indices:
            continue

        # Initially add the current point to the list of sampled indices
        sampled_indices.append(i)
        indices = kdtree.query_ball_point(coordinates[i], min_distance_threshold)

        # Track if an opposite COG point has been kept
        opposite_cog_point_kept = False

        for j in indices:
            if j != i:
                cog_diff = calculate_cog_difference(trajectories[i][4], trajectories[j][4])  # Assuming COG is at index 5
                if cog_diff > 160 and cog_diff < 205:  # COG difference approximately 180 degrees
                    if not opposite_cog_point_kept:  # Check if no opposite COG point has been kept yet
                        sampled_indices.append(j)
                        opposite_cog_point_kept = True

        # Update excluded indices to include all points within the radius, ensuring each is only considered once
        excluded_indices.update(indices)

    # Filter the trajectories to only include sampled points
    sampled_trajectories = [trajectories[i] for i in sampled_indices]

    print("Number of points after sampling:", len(sampled_trajectories))
    return sampled_trajectories


def create_nodes(sampled_trajectories, grid_size):
    print("Creating Nodes")
    """
        Receives the geometrically sampled AIS points, assigns them with a avg_depth
        by intersecting with a grid layer. The function then returns the a graph
        containing the nodes with relevant attributes. 
    """
    ais_points_gdf = gpd.GeoDataFrame(
        sampled_trajectories, 
        columns=['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type'],
        geometry=gpd.points_from_xy([item[1] for item in sampled_trajectories], [item[0] for item in sampled_trajectories]),
        crs='EPSG:4326'
        )

    config = load_config()
    engine = create_engine(f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    grid = f"SELECT * FROM {grid_size}"
    grids_gdf = gpd.read_postgis(
        grid, 
        engine, 
        geom_col='geometry'
        )
    
    join = gpd.sjoin(ais_points_gdf, grids_gdf, how='left', predicate='within')
    ais_points_gdf['avg_depth'] = join['avg_depth']

    G = nx.Graph()
    for index, row in ais_points_gdf.iterrows():
        node = (row['latitude'], row['longitude'])
        attributes = row.drop(['geometry']).to_dict()
        G.add_node(node, **attributes)

    return G


def create_edges(G, edge_radius_threshold, bearing_threshold, nodes_file_path, edges_file_path):
    print("Creating Edges")
    """
        Creates edges between nodes in the graph. The creation of an edge 
        is based on two criterias. Distance between nodes, and bearing between nodes.
    """
    node_coords_list = [(node, data) for node, data in G.nodes(data=True)]
    node_array = np.array([node for node, data in node_coords_list])
    
    kdtree = cKDTree(node_array)
    edge_count = 0

    for i, (node, data) in enumerate(node_coords_list):
        node_cog = data['cog']
        nearby_indices = kdtree.query_ball_point(node, edge_radius_threshold)
        
        for nearby_index in nearby_indices:
            if nearby_index != i: 
                nearby_node, nearby_data = node_coords_list[nearby_index]
                nearby_cog = nearby_data['cog']

                cog_diff = calculate_bearing_difference(node_cog, nearby_cog)
                
                if cog_diff <= bearing_threshold:
                    distance = haversine_distance(node[0], node[1], nearby_node[0], nearby_node[1])
                    G.add_edge(node, nearby_node, weight=distance)
                    edge_count += 1

    print(f"Total edges created: {edge_count}")

    export_graph_to_geojson(G, nodes_file_path, edges_file_path)

def create_graph(graph_trajectories, geometric_parameter, grid_size, edge_conneciton, bearing_parameter, nodes_file_path, edges_file_path):
    geometric_sampled_nodes = geometric_sampling(graph_trajectories, geometric_parameter)
    nodes = create_nodes(geometric_sampled_nodes, grid_size)
    create_edges(nodes, edge_conneciton, bearing_parameter, nodes_file_path, edges_file_path) 

def create_graphs_for_cells():
    all_trajectories = []
    
    for folder_name in os.listdir(INPUT_FOLDER_PATH):
        folder_path = os.path.join(INPUT_FOLDER_PATH, folder_name)

        if os.path.isdir(folder_path):
            print(f"Processing {folder_name}")
            trajectories = extract_original_trajectories(folder_path)

            if len(trajectories) == 0:
                    print(f"No trajectories found in {folder_name}, skipping...")
                    continue
            
            all_trajectories.extend(trajectories)

            output_subfolder = os.path.join(OUTPUT_FOLDER_PATH, folder_name)

            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

    nodes_file_path = os.path.join(output_subfolder, 'nodes.geojson')
    edges_file_path = os.path.join(output_subfolder, 'edges.geojson')

    print(len(all_trajectories))

    # if len(trajectories) > 2000000:
    #     create_graph(trajectories, 0.002, 'grid_400', 0.008, 45, nodes_file_path, edges_file_path)
    # if len(trajectories) > 1000000:
    #     create_graph(trajectories, 0.002, 'grid_400', 0.008, 45, nodes_file_path, edges_file_path)
    # if len(trajectories) > 500000:
    #     create_graph(trajectories, 0.0005, 'grid_400', 0.0015, 45, nodes_file_path, edges_file_path)
    # else:
    #     create_graph(trajectories, 0.00025, 'grid_400', 0.00075, 45, nodes_file_path, edges_file_path)
            

create_graphs_for_cells()


# def create_all_graphs():
#     graph_trajectories = extract_original_trajectories()
#     create_graph(graph_trajectories, 0.00025, 'grid_200', 0.00075, 45)



# def create_grid_layer(cell_size_km, output_file_path):
#     min_lat, min_lon, max_lat, max_lon = 53.00, 3.00, 59.00, 16.00
#     lat_conversion = 111.32  # km per degree
#     avg_lat = np.mean([min_lat, max_lat])
#     lon_conversion = lat_conversion * np.cos(np.radians(avg_lat))
    
#     lat_step = cell_size_km / lat_conversion
#     lon_step = cell_size_km / lon_conversion

#     n_cells_lat = int((max_lat - min_lat) / lat_step)
#     n_cells_lon = int((max_lon - min_lon) / lon_step)

#     with open(output_file_path, 'w') as file:
#         file.write("cell_id,min_lon,max_lon,min_lat,max_lat\n")
#         for i in range(n_cells_lat):
#             for j in range(n_cells_lon):
#                 lat0 = min_lat + i * lat_step
#                 lon0 = min_lon + j * lon_step
#                 lat1 = lat0 + lat_step
#                 lon1 = lon0 + lon_step

#                 cell_name = f"{i+1}_{j+1}"

#                 # Write each bounding box to the text file
#                 file.write(f"{cell_name},{lon0},{lon1},{lat0},{lat1}\n")

#     print(f"Grid data written to {output_file_path}")

# create_grid_layer(50.0, os.path.join(OUTPUT_FOLDER_PATH, f'graph_cargo/file.txt'))


#create_all_graphs()
