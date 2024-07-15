import numpy as np
import hou

""" To use this script in Houdini:
1. Create a .hip (or .hipnc) file in Houdini. 
2. Create a Geometry node using the /obj Network view in the GUI.
3. In the newly created geometry node, create a Python SOP module
4. In the Python SOP's "Python" parameter, copy and paste the code below. 
5. Change the user-defined parameters in the code to match the file path of the .npz file you want to visualize."""

def load_positions_and_types(file_path):
    data = np.load(file_path, allow_pickle=True)
    key = list(data.keys())[0]  # assuming the key is the first one in the file
    positions = data[key][0]  # Extract positions
    particle_types = data[key][1]  # Extract particle types
    return positions, particle_types  # Return positions and types

def update_positions(geo, positions, particle_types, timestep):
    timestep = int(timestep) # Convert timestep from a float to an integer
    # Clear existing geometry
    geo.clear()

    # Separate kinematic particles
    kinematic_indices = np.where(particle_types != 3)[0]
    print("Kinematic Indices Shape:", kinematic_indices.shape)
    print("Positions Shape:", positions.shape)
    kinematic_positions = positions[timestep][kinematic_indices]
    print("Kinematic Positions Shape:", kinematic_positions.shape)
        
    # Create points for kinematic particles
    kinematic_hou_points = [geo.createPoint() for _ in range(len(kinematic_positions))]
    for point, pos in zip(kinematic_hou_points, kinematic_positions):
        point.setPosition(pos.tolist())
        
    # Add the color attribute if it doesn't exist
    if not geo.findPointAttrib("Cd"):
        geo.addAttrib(hou.attribType.Point, "Cd", hou.Vector3(0, 0, 0))
    # Set color attribute for kinematic particles
    for point in kinematic_hou_points:
        point.setAttribValue("Cd", hou.Vector3(0, 0, 1))  # Set color to blue
        
# User-defined parameters
file_path = "/Users/Jonathan_1/Downloads/NHERI REU/GNS/GNS_1M_Water_Barrier/Test_Unfair/trajectory0.npz"
positions, particle_types = load_positions_and_types(file_path)

# Houdini Python SOP default code
node = hou.pwd()
geo = node.geometry()

# Get the current frame number and map it to the corresponding timestep
frame = hou.frame()
timestep = min(frame - 1, positions.shape[0] - 1)  # Ensure timestep is within bounds
print(f"This is the current timestep: {timestep}") # Print the current timestep to Houdini Python Shell
# Update positions based on the current timestep
update_positions(geo, positions, particle_types, timestep)
