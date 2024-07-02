from datahandler import TrajectoryDataHandler
import json
import argparse
import os
import sys

# Add the taichi_mpm_water directory to the system path
sys.path.append('/scratch/10029/jgaucin/gns-mpm-ls6/taichi_mpm_water') # Example

from run_mpm import run_collision

# Example usage
if __name__ == "__main__":
    """ 
    Args:
        input_path: Complete path to input.json file for Taichi MPM   
    """
    ## Parse through command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="temp_mpm_input.json", type=str, help="Input json file name")
    parser.add_argument('--material_feature', default=False, type=bool, help="Whether to add material property to node feature")
    # parser.add_argument('--output_dir', default="metadataoutput.json", type=str, help="Output directory to place .npz files") # Not needed
    args = parser.parse_args()
    input_path = args.input_path
    
    # Load the json file
    f = open(input_path)
    inputs = json.load(f)
    output_dir = inputs['save_path'] # Made the save path the output directory
    f.close()
    
    ## Create the dataset with MPM
    
    # save input file being used.
    if not os.path.exists(inputs['save_path']):
        os.makedirs(inputs['save_path'])
    input_filename = input_path.rsplit('/', 1)[-1]
    with open(f"{inputs['save_path']}/{input_filename}", "w") as input_file:
        json.dump(inputs, input_file, indent=4)

    for i in range(inputs["id_range"][0], inputs["id_range"][1]):
        data = run_collision(i, inputs, True, args)
        
    # Output path to save .npz files
    npz_directory = output_dir
    # Initialize instance of data handler using base directory {npz_directory}
    handler = TrajectoryDataHandler(npz_directory)

    # Inspect a specific .npz file
    # handler.inspect_npz_file("test.npz")

    # Merge trajectories (multiple .npz files) into a single .npz file
    handler.merge_trajectories("train.npz", num_trajectories=100)
    
    # Inspect the newly merged and created train.npz file
    handler.inspect_npz_file("train.npz")

    # Calculate and create metadata.json file
    handler.create_metadata("metadata", num_trajectories=100)
    # Notes: Necessary metadata (9):
    # {"bounds","sequence_length", "default_connectivity_radius","dim", "dt", "vel_mean", "vel_std", "acc_mean", "acc_std"}
    # Note: In GNS train.py sequence_length not necessary but recommended and dt is not necessary, only describing MPM simulation timesteps.