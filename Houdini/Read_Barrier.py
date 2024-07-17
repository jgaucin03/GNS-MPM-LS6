import numpy as np
import hou

""" To use this script in Houdini:
1. Create a .hip (or .hipnc) file in Houdini. 
2. Create a Geometry node using the /obj Network view in the GUI.
3. In the newly created geometry node, create a Python SOP module
4. In the Python SOP's "Python" parameter, copy and paste the code below. 
5. Change the user-defined parameters in the code to match the cube-like obstacles you want to visualize."""

def load_positions_and_types(file_path, timestep):
    data = np.load(file_path, allow_pickle=True)
    key = list(data.keys())[0]  # assuming the key is the first one in the file
    positions = data[key][0]  # Extract positions
    particle_types = data[key][1]  # Extract particle types
    return positions[timestep], particle_types  # Return positions and types at the specified timestep

def create_box(geo, center, size):
    # Create a box given the center and size
    x, y, z = center
    dx, dy, dz = size
    points = [
        geo.createPoint(), geo.createPoint(), geo.createPoint(), geo.createPoint(),
        geo.createPoint(), geo.createPoint(), geo.createPoint(), geo.createPoint()
    ]
    points[0].setPosition([x - dx, y - dy, z - dz])
    points[1].setPosition([x + dx, y - dy, z - dz])
    points[2].setPosition([x + dx, y + dy, z - dz])
    points[3].setPosition([x - dx, y + dy, z - dz])
    points[4].setPosition([x - dx, y - dy, z + dz])
    points[5].setPosition([x + dx, y - dy, z + dz])
    points[6].setPosition([x + dx, y + dy, z + dz])
    points[7].setPosition([x - dx, y + dy, z + dz])

    # Create the 6 faces of the box
    faces = [
        [points[0], points[1], points[2], points[3]],
        [points[4], points[5], points[6], points[7]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[1], points[2], points[6], points[5]],
        [points[4], points[7], points[3], points[0]]
    ]
    for face in faces:
        poly = geo.createPolygon()
        for point in face:
            poly.addVertex(point)

# User-defined parameters
file_path = "/Users/Jonathan_1/Downloads/NHERI REU/GNS/GNS_1M_Water_Barrier/Test_Unfair/trajectory0.npz"
timestep = 0  # These are stationary particles, so we can create their shape from any timestep and the shape will transfer

obstacles = [
    [1.7556741351888356, 0.5463226309780378, 1.2776900257563668, 0.1, 0.35, 0.1],
    [1.3199999274533551, 0.501869016571738, 0.12122617332311873, 0.1, 0.35, 0.1],
    [1.465502135320373, 0.3344900428700822, 0.8344616769242985, 0.1, 0.35, 0.1]
] # Rigid barrier data (initial positions and sizes) from particleinfo.json file. Can be loaded from a file if needed

# Load trajectory data for the specified timestep
# positions, particle_types = load_positions_and_types(file_path, timestep)



# Houdini Python SOP default code
node = hou.pwd()
geo = node.geometry()

# Clear existing geometry
geo.clear()

# Create boxes for each rigid barrier
for obstacle in obstacles:
    center = obstacle[:3]
    size = obstacle[3:]
    create_box(geo, center, size)
