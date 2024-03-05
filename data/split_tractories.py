import pandas as pd
import geopandas as gpd
from collections import namedtuple
from enum import Enum
from logs.logging import setup_logger

logging = setup_logger('trajectory_creation_log.txt')

class HarborState(Enum):
    STOPPING = 0
    LEAVING = 1
    AT_ANCHOR = 2
    PASSING_THROUGH = 3

def split_trajectories_from_df(harbors_df:gpd.GeoDataFrame, trajectory_df:gpd.GeoDataFrame):
    trajectory_df = trajectory_df.reset_index(drop=True)
    trajectories_df = order_by_diff_vessels(trajectory_df)
    trajectories_df = trajectories_df.drop_duplicates(subset=['vessel_id', 'timestamp'], keep='first')

    trajectories_df = trajectories_df.reset_index(drop=True) # to ensure indexes are still fine
    sub_trajectories_df = split_to_sub_trajectories_using_harbor(harbors_df, trajectories_df)
    
    return sub_trajectories_df

def order_by_diff_vessels(sorted_locations_df: gpd.GeoDataFrame):
    sorted_locations_df.fillna({'imo': -1, 'ship_type': 'None', 'width': -1, 'length': -1}, inplace=True)
    sorted_locations_df['vessel_id'] = sorted_locations_df.groupby(['imo', 'ship_type', 'width', 'length']).ngroup() 
    return sorted_locations_df

def split_to_sub_trajectories_using_harbor(harbors_df: gpd.GeoDataFrame, trajectories_df: gpd.GeoDataFrame):
    trajectories_df = add_in_harbor_column(trajectories_df, harbors_df)
    
    sub_trajectories = []
    
    try:
        for _, positions_df in trajectories_df.groupby('vessel_id'):
            trajectory_in_harbor = []
            current_sub_trajectory = []
            prev_position = None
            from_sea = False
            from_harbor = False
            is_stationary = False
            stationary_trajectory = []  
                
            for _, current_position in positions_df.iterrows(): 
                # find out if the sub trajectory starts by sea or harbor           
                if (prev_position is None):
                    from_harbor = current_position.in_harbor
                    from_sea = not from_harbor
                else:
                    if prev_position.geometry.equals(current_position.geometry):
                        prev_position = current_position
                
                is_stationary = is_vessel_stationary(current_position.navigational_status) and not current_position.in_harbor
                
                # choices, given we arrive by sea to harbor   
                if from_sea:
                    if is_stationary or stationary_trajectory:
                        (current_sub_trajectory, prev_position, stationary_trajectory) = handle_stationary_at_sea(current_sub_trajectory, current_position, is_stationary, stationary_trajectory)
                    elif current_position.in_harbor:
                        # trajectory is now in harbor. Keep track until we leave the harbor
                        trajectory_in_harbor, prev_position = handle_from_sea_to_harbor(trajectory_in_harbor, current_position)                
                    else:
                        if trajectory_in_harbor:
                            sub_trajectories, current_sub_trajectory, trajectory_in_harbor, prev_position, from_harbor, from_sea = handle_from_sea_left_harbor(
                                sub_trajectories, current_sub_trajectory, trajectory_in_harbor, current_position, from_harbor, from_sea)
                        else:
                            # trajectory is at sea. Keep track of sea location until we enter a harbor
                            current_sub_trajectory, prev_position = handle_from_sea_to_sea(current_sub_trajectory, current_position) 
                elif from_harbor:
                    if current_position.in_harbor:
                        trajectory_in_harbor, prev_position = handle_from_harbor_to_harbor(trajectory_in_harbor, current_position)
                    else:
                        sub_trajectories, current_sub_trajectory, trajectory_in_harbor, prev_position, from_harbor, from_sea = handle_from_harbor_to_sea(
                                sub_trajectories, current_sub_trajectory, trajectory_in_harbor, current_position, from_harbor, from_sea)
                else:
                    current_sub_trajectory.append(current_position)
                    prev_position = current_position
                    
            # add the last sub trajectory. We do not handle stationary, as that would have ben handled ealier
            if current_sub_trajectory:
                if trajectory_in_harbor:
                    current_sub_trajectory, trajectory_in_harbor = handle_last_sub_trajectory_entering_harbor(current_sub_trajectory, trajectory_in_harbor)
                
                sub_trajectories.append(current_sub_trajectory)

            elif trajectory_in_harbor: 
                current_sub_trajectory, trajectory_in_harbor = handle_last_trajectory_in_harbor(current_sub_trajectory, trajectory_in_harbor) 

                if current_sub_trajectory:
                    sub_trajectories.append(current_sub_trajectory)
                
        
        # Flatten the list of lists
        flattened_sub_trajectories = [
            {'sub_trajectory_id': i, **point} 
            for i, sub_trajectory in enumerate(sub_trajectories, start=1) 
            for point in sub_trajectory
        ]
            
        # Convert to DataFrame
        sub_trajectory_df = pd.DataFrame(flattened_sub_trajectories)
        return sub_trajectory_df
    
    except Exception as e:
        logging.error(f'Failed to create trajectories with error {repr(e)}')        

def add_in_harbor_column(trajectory_df:gpd.GeoDataFrame, harbors_df: gpd.GeoDataFrame):
    """
    Perform a spatial join between trajectory points and harbor polygons,
    and add a new column indicating whether each point is within a harbor.

    Parameters:
    trajectory_df (GeoPandas GeoDataFrame): GeoDataFrame containing trajectory points.
    harbors_df (GeoPandas GeoDataFrame): GeoDataFrame containing harbor polygons.

    Returns:
    GeoPandas GeoDataFrame: New GeoDataFrame with an additional 'in_harbor' column.
    """
    # Perform a spatial join to identify points from trajectory_df that intersect harbors
    points_with_harbors = gpd.sjoin(trajectory_df, harbors_df, how="left", predicate="intersects", lsuffix='left', rsuffix='right')

    # Create a new column 'in_harbor' with boolean values indicating whether each point is within a harbor
    points_with_harbors['in_harbor'] = ~points_with_harbors['index_right'].isnull()

    # Optional: Drop unnecessary columns from the resulting DataFrame
    points_with_harbors.drop(columns=['index_right'], inplace=True)

    return points_with_harbors

def handle_from_sea_to_harbor(trajectory_in_harbor, current_position):
    trajectory_in_harbor.append(current_position)
    prev_position = current_position
    return trajectory_in_harbor, prev_position

def is_vessel_stationary(navigational_status:str) -> bool:
    stationary_status = ['moored', 'at anchor']
    navigational_status = navigational_status.lower()
    return navigational_status in stationary_status

def handle_stationary_at_sea(current_sub_trajectory, current_position, is_stationary, stationary_trajectory):
    if (is_stationary):
        stationary_trajectory.append(current_position)
    
    prev_position = current_position
    
    if len(stationary_trajectory) == 1:
        return (current_sub_trajectory, prev_position, stationary_trajectory)
    
    moved_distance = stationary_trajectory[0].geometry.distance(current_position.geometry)
    
    if moved_distance > 25:
        current_sub_trajectory.extend(stationary_trajectory)
        current_sub_trajectory.append(current_position)
        stationary_trajectory = []
        return (current_sub_trajectory, prev_position, stationary_trajectory)

    stationary_trajectory.append(current_position)
    return (current_sub_trajectory, prev_position, stationary_trajectory)

    

def handle_from_sea_left_harbor(sub_trajectories, current_sub_trajectory, trajectory_in_harbor, current_position, from_harbor, from_sea):
    (state, to_current_sub_trajectory, for_next_sub_trajectory) = get_harbor_state_when_entering_harbor(trajectory_in_harbor)              
   
    if (state == HarborState.PASSING_THROUGH):
        current_sub_trajectory.extend(to_current_sub_trajectory)
        current_sub_trajectory.append(current_position)
        trajectory_in_harbor = []
        prev_position = current_position
        
        return sub_trajectories, current_sub_trajectory, trajectory_in_harbor, prev_position, from_harbor, from_sea
        
    elif (state == HarborState.STOPPING):
        current_sub_trajectory.extend(to_current_sub_trajectory)
        sub_trajectories.append(current_sub_trajectory)
        current_sub_trajectory = []
        current_sub_trajectory.append(current_position)
        trajectory_in_harbor = for_next_sub_trajectory

        if (trajectory_in_harbor):
            (state, in_current_sub_trajectory) = get_harbor_state_when_leaving_harbor(trajectory_in_harbor) 
            if state == HarborState.LEAVING:
                in_current_sub_trajectory.extend(current_sub_trajectory)
                current_sub_trajectory = []
                current_sub_trajectory = in_current_sub_trajectory
            elif state == HarborState.AT_ANCHOR:
                trajectory_in_harbor = []
            else:
                logging.error('State was neither leaving or at anchor')
        
        from_harbor = current_position.in_harbor
        from_sea = not from_harbor
            
    prev_position = current_position
    
    return sub_trajectories, current_sub_trajectory, trajectory_in_harbor, prev_position, from_harbor, from_sea

def handle_from_sea_to_sea(current_sub_trajectory, current_position):
    current_sub_trajectory.append(current_position)
    prev_position = current_position
    return current_sub_trajectory, prev_position

def handle_from_harbor_to_harbor(trajectory_in_harbor, current_position):
    trajectory_in_harbor.append(current_position)
    prev_position = current_position
    
    return trajectory_in_harbor, prev_position

def handle_from_harbor_to_sea(sub_trajectories, current_sub_trajectory, trajectory_in_harbor, current_position, from_harbor, from_sea):
    (state, in_current_sub_trajectory) = get_harbor_state_when_leaving_harbor(trajectory_in_harbor) 
    
    if (state == HarborState.LEAVING):
        current_sub_trajectory.extend(in_current_sub_trajectory)
        current_sub_trajectory.append(current_position)
        trajectory_in_harbor = []
    else:
        if (state == HarborState.AT_ANCHOR): # don't do anything
            trajectory_in_harbor = []
    
    prev_position = current_position
    from_harbor = current_position.in_harbor
    from_sea = not from_harbor
            
    return sub_trajectories, current_sub_trajectory, trajectory_in_harbor, prev_position, from_harbor, from_sea

def handle_last_sub_trajectory_entering_harbor(current_sub_trajectory, trajectory_in_harbor):
    (state, to_current_sub_trajectory, for_next_sub_trajectory) = get_harbor_state_when_entering_harbor(trajectory_in_harbor)
    
    if (state == HarborState.PASSING_THROUGH):
        current_sub_trajectory.extend(to_current_sub_trajectory)
    elif (state == HarborState.STOPPING):
        current_sub_trajectory.extend(to_current_sub_trajectory)
    
    trajectory_in_harbor = for_next_sub_trajectory
        
    if (trajectory_in_harbor):
        (state, in_current_sub_trajectory) = get_harbor_state_when_leaving_harbor(trajectory_in_harbor) 
        if state == HarborState.LEAVING:
            in_current_sub_trajectory.extend(current_sub_trajectory)
            current_sub_trajectory = []
            current_sub_trajectory = in_current_sub_trajectory
        elif state == HarborState.AT_ANCHOR:
            trajectory_in_harbor = []
        else:
            logging.error('State was neither leaving or at anchor')
    
    return current_sub_trajectory, for_next_sub_trajectory

def handle_last_trajectory_in_harbor(current_sub_trajectory, trajectory_in_harbor):
    (state, in_current_sub_trajectory) = get_harbor_state_when_leaving_harbor(trajectory_in_harbor)     
    if (state == HarborState.LEAVING):
        current_sub_trajectory.extend(in_current_sub_trajectory)
        trajectory_in_harbor = []
        
    else:
        if (state == HarborState.AT_ANCHOR): # don't do anything
            trajectory_in_harbor = []
    
    return current_sub_trajectory, trajectory_in_harbor

def get_harbor_state_when_entering_harbor(in_harbor_positions:list) -> tuple[HarborState, list, list]:
    EnteringTuple = namedtuple("EnteringTuple", ['harbor_state', 'to_current_sub_trajectory', 'for_next_sub_strajectory'])
    state:HarborState
    
    if len(in_harbor_positions) >= 2:
        return get_entering_harbor_positions(in_harbor_positions)        
    else:
        return EnteringTuple(harbor_state=HarborState.PASSING_THROUGH, to_current_sub_trajectory=in_harbor_positions, for_next_sub_strajectory=[])

def get_entering_harbor_positions(in_harbor_positions:list) -> tuple[HarborState, list, list]:
    to_current_sub_trajectory = []
    for_next_sub_trajectory = []
    
    EnteringTuple = namedtuple("EnteringTuple", ['harbor_state', 'to_current_sub_trajectory', 'for_next_sub_trajectory'])
    still_indexes:list = [index for index, current_position in enumerate(in_harbor_positions) if current_position.sog == 0.0]
        
  
    if not still_indexes:
        to_current_sub_trajectory = in_harbor_positions
        return EnteringTuple(harbor_state=HarborState.PASSING_THROUGH, to_current_sub_trajectory=to_current_sub_trajectory, for_next_sub_trajectory=for_next_sub_trajectory)    
    
    stopped_index = still_indexes[0] + 1 # includes stoped position
    moved_index = still_indexes[-1] + 1 # start from position we move
    to_current_sub_trajectory = in_harbor_positions[:stopped_index]
    for_next_sub_trajectory = in_harbor_positions[moved_index:]
    
    return EnteringTuple(harbor_state=HarborState.STOPPING, to_current_sub_trajectory=to_current_sub_trajectory, for_next_sub_trajectory=for_next_sub_trajectory)

def get_harbor_state_when_leaving_harbor(in_harbor_positions:list) -> tuple[HarborState, list]:
    LeavingTuple = namedtuple("LeavingTuple", ['harbor_state', 'to_current_sub_trajectory'])
    
    if len(in_harbor_positions) > 2:
        return get_leaving_harbor_positions(in_harbor_positions)
    else:
        return LeavingTuple(harbor_state=HarborState.AT_ANCHOR, to_current_sub_trajectory=[]) 

def get_leaving_harbor_positions(harbor_positions:list) -> tuple[HarborState, list]:   
    LeavingTuple = namedtuple("LeavingTuple", ['harbor_state', 'to_current_sub_trajectory'])
    first_position = harbor_positions[0]
    
    stationary_indexes = [index for index, current_position in enumerate(harbor_positions) if is_vessel_stationary(current_position.navigational_status)]
        
    if (stationary_indexes):
        if len(stationary_indexes) == len(harbor_positions):
            return LeavingTuple(harbor_state=HarborState.AT_ANCHOR, to_current_sub_trajectory=[])
    
        if stationary_indexes[-1] != (len(harbor_positions) - 1):
            leaving_index = stationary_indexes[-1] + 1
            to_current_sub_trajectory = harbor_positions[leaving_index:]
            return LeavingTuple(harbor_state=HarborState.LEAVING, to_current_sub_trajectory=to_current_sub_trajectory)
        else:
            return LeavingTuple(harbor_state=HarborState.AT_ANCHOR, to_current_sub_trajectory=[])

    # if no stationary navigational status was found
    is_leaving = next((True for current_position in harbor_positions[1:] if first_position.geometry.distance(current_position.geometry) > 10), False)    
    
    if (is_leaving):
        still_indexes:list = [index for index, current_position in enumerate(harbor_positions) if current_position.sog == 0.0] 
        last_still_index = still_indexes[-1] + 1 if still_indexes else 0
        leaving_positions = harbor_positions[last_still_index:]
        
        return LeavingTuple(harbor_state=HarborState.LEAVING, to_current_sub_trajectory=leaving_positions)
    else:
        return LeavingTuple(harbor_state=HarborState.AT_ANCHOR, to_current_sub_trajectory=[])