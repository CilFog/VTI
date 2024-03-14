import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import rasterio
from pyproj import Transformer, CRS
from shapely.geometry import Point, box
import geopandas as gpd
import geoalchemy2
import sqlalchemy
from sqlalchemy import create_engine, ForeignKey
from config import load_config
from connect import connect

DEPTH_MAP = 'C:/Users/alexf/Desktop/ddm_50m.dybde.tiff'
#DEPTH_MAP = '/srv/P-10/ddm_50m.dybde.tiff'

def extract_depth_map():
    points = []
    processed_pixels = 0
    pixels_to_reach = 1000000

    print("Starting depth map extraction")

    with rasterio.open(DEPTH_MAP, crs="3034") as dataset:
        
        # Select only pixels with value:
        msk = dataset.read_masks(1)

        # Get geotransform information
        transform = dataset.transform

        # Get image size
        width = dataset.width
        height = dataset.height

        # Read the depth values
        depths = dataset.read(1)

        transformer = Transformer.from_crs(CRS.from_epsg(3034), CRS.from_epsg(4326))

        # Iterate through all pixels and print their corresponding latitudes and longitudes
        for row in range(height):
            for col in range(width):
                if msk[row, col] > 0:
                    # Calculate latitude and longitude for each pixel
                    lat, lon = transform * (col + 0.5, row + 0.5)
                
                    lon_trans, lat_trans = transformer.transform(lon, lat)

                    depth_value = depths[row, col]

                    point = Point(lat_trans, lon_trans)
                    points.append((point, depth_value))
                    processed_pixels += 1

                    if processed_pixels % pixels_to_reach == 0:
                        print(f"Reached {processed_pixels} processed pixels.")

                if processed_pixels == 10000:
                        break
            if processed_pixels == 10000:
                break

    gdf = gpd.GeoDataFrame(geometry=[point for point, _ in points], crs="EPSG:4326")
    gdf['depth'] = [depth for _, depth in points]


    create_and_insert_grid_into_db(gdf, 1.6, '1600')
    create_and_insert_grid_into_db_with_relation(gdf, 0.8, '800', 'grid_1600')
    #create_and_insert_grid_into_db_with_relation(gdf, 0.4, '400', 'grid_800')
    #create_and_insert_grid_into_db_with_relation(gdf, 0.2, '200', 'grid_400')
    #create_and_insert_grid_into_db_with_relation(gdf, 0.1, '100', 'grid_200')
    #create_and_insert_grid_into_db_with_relation(gdf, 0.05, '50', 'grid_100')
    #create_and_insert_grid_into_db_with_relation(gdf, 0.025, '25', 'grid_50')


def create_and_insert_grid_into_db(gdf, cell_size_km, table_name):

    grid_gdf = create_grid_layer(cell_size_km)

    # Spatial join
    joined = gpd.sjoin(gdf, grid_gdf, how='right', op='within')
    valid_joined = joined.dropna(subset=['index_left', 'depth'])
    grouped_joined = valid_joined.groupby('geometry')['depth'].mean().reset_index()
    grouped_joined = gpd.GeoDataFrame(grouped_joined, geometry='geometry')

    grid_gdf = grouped_joined

    try:
        # Load configuration from the config file
        config = load_config()

        engine = create_engine('postgresql://'+config['user']+':'+config['password']+'@'+config['host']+':'+config['port']+'/'+config['database']+'')

        # Connect to the PostgreSQL database
        conn = connect(config)

        # Create a cursor object using the connection
        cursor = conn.cursor()

        try:
            print("Inserting into db")

            # Convert geometries to WKT
            grid_gdf['geometry_wkt'] = grid_gdf['geometry'].apply(lambda geom: geom.wkt)
            grid_gdf = grid_gdf.drop(columns=['geometry'])

            grid_gdf[['geometry_wkt', 'depth']].reset_index(drop=True).to_sql(
                f'grid_{table_name}', 
                engine, 
                index=True, 
                if_exists='replace', 
                dtype={
                'geometry_wkt': geoalchemy2.Geometry('POLYGON', srid=4326),
                'depth': sqlalchemy.types.Float(),
            })

        except psycopg2.Error as e:
            print("Error executing SQL command:", e)

        finally:
            print("Inserting into db was succesfully")
            cursor.close()
        

    except Exception as e:
        print(e)

    finally:
        cursor.close()
        conn.close()

def create_and_insert_grid_into_db_with_relation(gdf, cell_size_km, table_name, parent_grid):

    grid_gdf = create_grid_layer(cell_size_km)

    # Spatial join
    joined = gpd.sjoin(gdf, grid_gdf, how='right', op='within')
    valid_joined = joined.dropna(subset=['index_left', 'depth'])
    grouped_joined = valid_joined.groupby('geometry')['depth'].mean().reset_index()
    grouped_joined = gpd.GeoDataFrame(grouped_joined, geometry='geometry')

    grid_gdf = grouped_joined

    try:
        # Load configuration from the config file
        config = load_config()

        engine = create_engine('postgresql://'+config['user']+':'+config['password']+'@'+config['host']+':'+config['port']+'/'+config['database']+'')

        # Connect to the PostgreSQL database
        conn = connect(config)

        # Create a cursor object using the connection
        cursor = conn.cursor()

        try:
            print("Creating key constraint between grids, and inserting into db")

            # Retrieve the existing grid_1000 data from the database
            parent_grid_id = f"SELECT * FROM {parent_grid}"
            parent_grid_id_gdf = gpd.read_postgis(parent_grid_id, engine, geom_col='geometry_wkt')

            # Create a new column 'parent_grid_id' in grid_gdf and populate with parent_grid_id if it is contianed within it
            grid_gdf['parent_grid_id'] = None
            grid_gdf['parent_grid_id'] = grid_gdf.apply(lambda row: find_parent_grid(row['geometry'], parent_grid_id_gdf), axis=1)


            # Convert geometries to WKT
            grid_gdf['geometry_wkt'] = grid_gdf['geometry'].apply(lambda geom: geom.wkt)
            grid_gdf = grid_gdf.drop(columns=['geometry'])

            # Write to the database with the foreign key constraint
            grid_gdf[['geometry_wkt', 'depth', 'parent_grid_id']].reset_index(drop=True).to_sql(
                f'grid_{table_name}', 
                engine,
                index=True,  
                if_exists='replace',
                dtype={
                    'geometry_wkt': geoalchemy2.Geometry('POLYGON', srid=4326),
                    'depth': sqlalchemy.types.Float(),
                    'parent_grid_id': sqlalchemy.types.Integer(),
                },
            )

        except psycopg2.Error as e:
            print("Error executing SQL command:", e)

        finally:
            print("Inserting into db was succesfully")
            cursor.close()
        

    except Exception as e:
        print(e)

    finally:
        cursor.close()
        conn.close()

def find_parent_grid(geometry, parent_grid):
    centroid = geometry.centroid
    intersecting_grid = parent_grid[parent_grid['geometry_wkt'].intersects(centroid)]
    
    if not intersecting_grid.empty:
        return intersecting_grid['index'].values[0]
    else:
        return None

def create_grid_layer(cell_size_km):
    min_lat, min_lon, max_lat, max_lon = 53.00, 7.00, 59.00, 17.00

    # Calculate the number of cells in the latitude and longitude directions
    n_cells_lat = int((max_lat - min_lat) / (cell_size_km / 111.32))  # Approximate conversion from km to degrees
    n_cells_lon = int((max_lon - min_lon) / (cell_size_km / (111.32 * np.cos(np.radians(np.mean([min_lat, max_lat]))))))

    lon_step = cell_size_km / (111.32 * np.cos(np.radians(np.mean([min_lat, max_lat]))))
    n_cells_lon = int((max_lon - min_lon) / lon_step)

    # Create grid cells
    grid_cells = []
    for i in range(n_cells_lat):
        for j in range(n_cells_lon):
            lat0 = min_lat + i * (cell_size_km / 111.32)  # Convert to degrees
            lon0 = min_lon + j * lon_step
            lat1 = lat0 + (cell_size_km / 111.32)  # Convert to degrees
            lon1 = lon0 + lon_step

            grid_cells.append(box(lon0, lat0, lon1, lat1))

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

    return grid_gdf

extract_depth_map()