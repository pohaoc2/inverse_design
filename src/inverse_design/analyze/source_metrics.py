# calculate the capillary density for a hexagonal grid-source layout

import numpy as np

MICRON_to_MM = 1e-3

def calculate_capillary_density(radius_bound, length_spacing, width_spacing, side_length):
    """
    Calculate capillary density for a hexagonal grid-source layout
    
    Parameters:
    radius_bound (float): Simulation radius + margin
    length_spacing (float): spacing in length dimension
    width_spacing (float): spacing in width dimension
    side_length (float): Length of one side of the triangular lattice (in microns)
    
    Returns:
    float: Capillary density (capillaries/mm^2)
    """
    # Calculate grid dimensions
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2
    
    # Calculate total number of capillaries (N)
    num_capillaries = (length // length_spacing) * (width // width_spacing)
    
    # Calculate area (A)
    area = ((length + 1) / 2 * side_length) * (width * side_length)
    
    # Calculate capillary density

    area = area * MICRON_to_MM**2
    density = num_capillaries / area
    
    return density

def calculate_distance_between_points(point1, point2, side_length, radius_bounds, depth_bounds=1, height_offset=0):
    """
    Calculate the distance between two points in a hexagonal grid.
    
    Args:
        point1: numpy array [x, y, z]
        point2: numpy array [x, y, z]
        side_length: float, side length of the hexagonal grid [microns]
    
    Returns:
        float, distance between the two points [mm]
    """
    
    u1, v1, w1, z1 = translate_xyz_to_uvwz(point1, radius_bounds, depth_bounds, height_offset)
    u2, v2, w2, z2 = translate_xyz_to_uvwz(point2, radius_bounds, depth_bounds, height_offset)
    difference = np.array([u1, v1, w1, z1]) - np.array([u2, v2, w2, z2])
    return np.sum(np.abs(difference)) / 2 * side_length * np.sqrt(3) * MICRON_to_MM

def translate_xyz_to_uvwz(coordinate, radius_bounds, depth_bounds=1, height_offset=0):
    """
    Translate XYZ coordinates to UVWZ coordinates.
    
    Args:
        coordinate: numpy array [x, y, z]
        radius_bounds: int, radius bounds
        depth_bounds: int, depth bounds (default=1)
        height_offset: int, height offset (default=0)
    
    Returns:
        numpy array [u, v, w, z] or None if out of bounds
    """
    
    x, y, z = coordinate
    z = z - depth_bounds + 1
    zo = abs(height_offset + z) % 3
    
    uu = (x - (-1 if zo == 2 else zo) + 2) / 3.0 - radius_bounds
    u = round(round(uu))
    
    vw = y - 2 * radius_bounds + 2 - (0 if zo == 0 else 1)
    v = -(vw + u) // 2 
    w = -(u + v)
    
    if abs(v) > radius_bounds or abs(w) > radius_bounds:
        return None
        
    return np.array([u, v, w, z])
