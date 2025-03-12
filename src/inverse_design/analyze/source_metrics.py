# calculate the capillary density for a hexagonal grid-source layout

def calculate_capillary_density(radius_bound, length_spacing, width_spacing, side_length):
    """
    Calculate capillary density for a hexagonal grid-source layout
    
    Parameters:
    radius_bound (float): Simulation radius + margin
    length_spacing (float): spacing in length dimension
    width_spacing (float): spacing in width dimension
    side_length (float): Length of one side of the triangular lattice
    
    Returns:
    float: Capillary density (capillaries per unit area)
    """
    # Calculate grid dimensions
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2
    
    # Calculate total number of capillaries (N)
    num_capillaries = (length // length_spacing) * (width // width_spacing)
    
    # Calculate area (A)
    area = ((length + 1) / 2) * (width * side_length)
    
    # Calculate capillary density
    density = num_capillaries / area
    
    return density
