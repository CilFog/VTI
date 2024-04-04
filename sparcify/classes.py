class SparsifyResult():
    reduced_points:int
    number_of_points:int
    trajectory_was_used:bool
    
    def __init__(self, reduced_points:int, number_of_points:int, trajectory_was_used:bool):
        self.reduced_points = reduced_points
        self.number_of_points = number_of_points
        self.trajectory_was_used = trajectory_was_used
