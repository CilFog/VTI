from shapely.geometry import Polygon, box

class SparsifyResult():
    reduced_points:int
    number_of_points:int
    trajectory_was_used:bool
    
    def __init__(self, reduced_points:int, number_of_points:int, trajectory_was_used:bool):
        self.reduced_points = reduced_points
        self.number_of_points = number_of_points
        self.trajectory_was_used = trajectory_was_used

brunsbuettel_to_kiel_polygon:Polygon = Polygon([(9.148469, 53.8867593),
                                                    (9.3086607, 53.9837313),
                                                    (9.3567259, 54.1312446),
                                                    (9.635504, 54.2076162),
                                                    (9.6629698, 54.2750273),
                                                    (9.7302611, 54.2854501),
                                                    (9.7357542, 54.3159018),
                                                    (9.8000041, 54.3547344),
                                                    (9.8528758, 54.3535339),
                                                    (9.9387065, 54.3291166),
                                                    (10.0300304, 54.3491318),
                                                    (10.0725492, 54.3519962),
                                                    (10.0986417, 54.3619995),
                                                    (10.1343473, 54.3575983),
                                                    (10.1254209, 54.308754),
                                                    (10.1803525, 54.306751),
                                                    (10.1899656, 54.3611993),
                                                    (10.0965818, 54.3783996),
                                                    (10.0519498, 54.3672002),
                                                    (10.0045644, 54.361078),
                                                    (9.9613126, 54.3491948),
                                                    (9.8693021, 54.3688003),
                                                    (9.7951444, 54.3680003),
                                                    (9.6961212, 54.3253338),
                                                    (9.7002411, 54.3081114),
                                                    (9.6330797, 54.2886346),
                                                    (9.5904394, 54.2273655),
                                                    (9.5286413, 54.1984551),
                                                    (9.3343206, 54.1470089),
                                                    (9.3174677, 54.1251252),
                                                    (9.2707758, 53.9897028),
                                                    (9.1285469, 53.8889118),
                                                    (9.148469, 53.8867593)])

aalborg_harbor_to_kattegat_bbox:Polygon = box(minx=9.841940, miny=56.970433, maxx=10.416415, maxy=57.098774)
doggersbank_to_lemvig_bbox:Polygon = box(minx=3.5, miny=54.5, maxx=8.5, maxy=56.5)
skagens_harbor_bbox:Polygon = box(minx=10.39820, miny=57.54558, maxx=11.14786, maxy=58.41927)

