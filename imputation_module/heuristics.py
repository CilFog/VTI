from math import radians, sin, cos, sqrt, atan2

def calculate_cog_divergence(cog1, cog2):
    divergence = abs(cog1 - cog2) % 360  # Normalize to [0, 360)
    if divergence > 180:  # Normalize to [0, 180]
        divergence = 360 - divergence
    return divergence

def heuristics(coord1, coord2):

    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c

    total_cost = distance

    return total_cost