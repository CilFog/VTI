from pyproj import CRS, Transformer
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import geoalchemy2
import sqlalchemy
from sqlalchemy import create_engine, Column, Float, Integer, ForeignKey, text, Index
from db_connection.config import load_config
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point, box
from geoalchemy2.shape import from_shape
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Float, Integer, ForeignKey

#DEPTH_MAP = 'C:/Users/alexf/Desktop/ddm_50m.dybde.tiff'
DEPTH_MAP = '/srv/P-10/ddm_50m.dybde.tiff'

# Assuming load_config is a function you've defined to load your database configuration
config = load_config()

# Database setup
engine_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
engine = create_engine(engine_url)
Base = declarative_base()


class DepthPoint(Base):
    __tablename__ = 'depth_points'
    id = Column(Integer, primary_key=True)
    geometry = Column(geoalchemy2.Geometry('POINT', srid=4326))
    depth = Column(Float)

# Create the tables in the database
Base.metadata.create_all(engine)

def insert_data_into_db_gdf(gdf):
    Session = sessionmaker(bind=engine)
    session = Session()

    for index, row in gdf.iterrows():
        wkt_element = geoalchemy2.WKTElement(row['geometry'].wkt, srid=4326)
        depth_point = DepthPoint(geometry=wkt_element, depth=row['depth'])
        session.add(depth_point)
    
    session.commit()
    session.close()

    print(f"Inserted batch of {len(gdf)} points into the database.")

def extract_depth_map():
    processed_pixels = 0
    batch_size = 100000

    print("Starting depth map extraction")

    with rasterio.open(DEPTH_MAP, crs="3034") as dataset:
        msk = dataset.read_masks(1)
        transform = dataset.transform
        width = dataset.width
        height = dataset.height
        depth_values = dataset.read(1)
        transformer = Transformer.from_crs(CRS.from_epsg(3034), CRS.from_epsg(4326))

        points = []
        depths = []

        # Iterate through all pixels and print their corresponding latitudes and longitudes
        for row in range(height):
            for col in range(width):
                if msk[row, col] > 0:
                    # Calculate latitude and longitude for each pixel
                    lat, lon = transform * (col + 0.5, row + 0.5)
                    lon_trans, lat_trans = transformer.transform(lon, lat)

                    # Append to temporary lists
                    points.append(Point(lat_trans, lon_trans))
                    depths.append(depth_values[row, col])

                    processed_pixels += 1

                    # Check if the batch size is reached
                    if len(points) == batch_size:
                        # Convert to GeoDataFrame and insert
                        gdf = gpd.GeoDataFrame({'geometry': points, 'depth': depths}, crs="EPSG:4326")
                        insert_data_into_db_gdf(gdf)
                        
                        # Clear the lists for the next batch
                        points.clear()
                        depths.clear()

        # Insert any remaining points after looping through all pixels
        if points:
            gdf = gpd.GeoDataFrame({'geometry': points, 'depth': depths}, crs="EPSG:4326")
            insert_data_into_db_gdf(gdf)

extract_depth_map()