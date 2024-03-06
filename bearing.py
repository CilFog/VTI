
import math
import numpy as np
import geopandas as gp

def calculate_initial_compass_bearing(df_curr:gp.GeoDataFrame, df_next:gp.GeoDataFrame) -> gp.GeoDataFrame:
    """
    Summary: Calculates the compas bearing between consecutive rows in a dataframe

    Args:
        df_curr (gp.GeoDataFrame): dataframe containing all first positions
        df_next (gp.GeoDataFrame): dataframe containing all next positions

    Returns:
        gp.GeoDataFrame: df_curr with a bearing
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

def get_haversine_dist_in_meters(df_curr:gp.GeoDataFrame, df_next:gp.GeoDataFrame) -> gp.GeoDataFrame:
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
    #dist.fillna(0, inplace=True)
    
    return dist