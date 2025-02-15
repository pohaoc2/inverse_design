import numpy as np
from scipy import ndimage
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def create_hex_grid(image_shape, hex_size):
    """Create hexagonal grid coordinates over the image."""
    height, width = image_shape
    
    # Calculate spacing between hexagon centers
    x_spacing = hex_size * 3/2
    y_spacing = hex_size * np.sqrt(3)
    
    # Create coordinate grids
    x_coords = np.arange(0, width, x_spacing)
    y_coords = np.arange(0, height, y_spacing)
    
    # Create offset for every other row
    x_offset = hex_size * 3/4
    
    # Generate all hexagon center coordinates
    centers = []
    for i, y in enumerate(y_coords):
        offset = x_offset if i % 2 else 0
        for x in x_coords:
            centers.append([x + offset, y])
            
    return np.array(centers)

def assign_pixels_to_cells(binary_image, hex_centers, hex_size):
    """Assign each occupied pixel to nearest hexagonal cell."""
    occupied_pixels = np.where(binary_image > 0)
    pixel_coords = np.column_stack((occupied_pixels[1], occupied_pixels[0]))  # x,y format
    
    # Find nearest hexagon center for each occupied pixel
    assignments = {}
    for pixel in pixel_coords:
        distances = np.linalg.norm(hex_centers - pixel, axis=1)
        nearest_center_idx = np.argmin(distances)
        center = tuple(hex_centers[nearest_center_idx])
        if center not in assignments:
            assignments[center] = []
        assignments[center].append(tuple(pixel))
    
    return assignments

def cube_to_axial(cube_coords):
    """Convert cube coordinates (u,v,w) to axial coordinates (q,r)."""
    q = cube_coords[0]
    r = cube_coords[2]
    return (q, r)

def axial_to_cube(axial_coords):
    """Convert axial coordinates (q,r) to cube coordinates (u,v,w)."""
    q, r = axial_coords
    u = q
    v = -q - r
    w = r
    return (u, v, w)

def get_symmetric_positions(u, v, w):
    """Get all symmetric positions for given cube coordinates."""
    return [
        (u, v, w),
        (-w, -u, -v),
        (v, w, u),
        (-u, -v, -w),
        (w, u, v),
        (-v, -w, -u)
    ]

def calculate_symmetry_index(binary_image, hex_size=10):
    """Calculate hexagonal symmetry index for binary image."""
    # Create hexagonal grid
    hex_centers = create_hex_grid(binary_image.shape, hex_size)
    
    # Assign pixels to hexagonal cells
    cell_assignments = assign_pixels_to_cells(binary_image, hex_centers, hex_size)
    
    # Convert occupied hexagon centers to cube coordinates
    occupied_cells = set()
    for center in cell_assignments.keys():
        # Convert pixel coordinates to cube coordinates
        # This is a simplified conversion - you may need to adjust based on your coordinate system
        q = round(center[0] / (hex_size * 3/2))
        r = round(center[1] / (hex_size * np.sqrt(3)) - (q / 2))
        cube_coords = axial_to_cube((q, r))
        occupied_cells.add(cube_coords)
    
    # Calculate symmetry index
    total_positions = 0
    total_occupied = 0
    
    # Check each occupied position and its symmetric counterparts
    checked_positions = set()
    for pos in occupied_cells:
        if pos in checked_positions:
            continue
            
        symmetric_positions = get_symmetric_positions(*pos)
        for sym_pos in symmetric_positions:
            if sym_pos not in checked_positions:
                total_positions += 1
                if sym_pos in occupied_cells:
                    total_occupied += 1
                checked_positions.add(sym_pos)
    
    symmetry_index = total_occupied / total_positions if total_positions > 0 else 0
    return symmetry_index

def load_isic_image(image_name, max_size=None):
    """Load ISIC segmentation image by ID."""
    try:
        binary_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        if binary_image is None:
            print(f"Warning: Could not load image: {image_name}")
            return None
        
        # Resize if max_size is specified
        if max_size is not None:
            height, width = binary_image.shape
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
                
            binary_image = cv2.resize(binary_image, (new_width, new_height), 
                                    interpolation=cv2.INTER_NEAREST)
            binary_image = (binary_image > 127).astype(np.uint8) * 255
        
        return binary_image
        
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        return None

def main():
    """Example usage of symmetry analysis."""
    # Get project root and construct data path
    data_path = "/home/pohaoc2/UW/bagherilab/inverse_design/data/ISIC2017/"
    all_images = os.listdir(data_path)
    all_images = [os.path.join(data_path, image_name) for image_name in all_images]

    size = 128  # Use smaller size for faster processing
    
    symmetry_indices = []
    if os.path.exists(data_path + 'symmetry_indices.pkl'):
        with open(data_path + 'symmetry_indices.pkl', 'rb') as f:
            symmetry_indices = pickle.load(f)
    else:
        for i, image_name in enumerate(all_images):
            binary_image = load_isic_image(image_name, size)
            if binary_image is None:
                continue
            symmetry_index = calculate_symmetry_index(binary_image)
            symmetry_indices.append(symmetry_index)
            if i % 100 == 0:
                print(f"Processed {i}/{len(all_images)} images")

        with open(data_path + 'symmetry_indices.pkl', 'wb') as f:
            pickle.dump(symmetry_indices, f)

    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=symmetry_indices)
    plt.title('Distribution of Symmetry Indices in ISIC Lesions')
    plt.ylabel('Symmetry Index')
    
    # Add statistical information
    plt.text(0.02, 0.98, 
             f'Mean: {np.mean(symmetry_indices):.3f}\n'
             f'Std: {np.std(symmetry_indices):.3f}\n'
             f'Median: {np.median(symmetry_indices):.3f}\n'
             f'N: {len(symmetry_indices)}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(data_path + 'symmetry_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mean symmetry index: {np.mean(symmetry_indices):.3f}")
    print(f"Std symmetry index: {np.std(symmetry_indices):.3f}")

if __name__ == "__main__":
    main()
