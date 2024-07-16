from datahandler import TrajectoryDataHandler
import json
import argparse
import os
import sys

# Add the taichi_mpm_water directory to the system path
sys.path.append('/scratch/10029/jgaucin/gns-mpm-ls6/taichi_mpm_water') # Example

from run_mpm import run_collision

# Test inputs are modified:
# "sim_space: [[0.1,1.9],[0.1,1.9], [0.1,1.9]] # Twice as large
# "nsteps": 500 # From 350
# "gen_cube_randomly": {
    # {"mass": {"ncubes": [2,5], "cube_gen_space": [[0.11, 1.01], [0.11, 1.89], [0.11, 1.89]], "nparticle_limits": 30000},
    # "obstacles": {"ncubes": [2, 5], "cube_gen_space": [[1.01, 1.89], [0.1, 1.05], [0.11, 1.89]] }} 
# TODO: Change cube gen space assumed 'z' dimension to only start at 0.1

# for i in range(4,10):
#    utils.animation_from_npz(path="/scratch/10029/jgaucin/gns-mpm-ls6/taichi_mpm_water/saved/", 
#    npz_name=f"trajectory{i}", save_name=f"trajectory{i}", 
#    boundaries=[[0.1,1.9],[0.1,1.9], [0.1,1.9]], timestep_stride=5, follow_taichi_coord=True)
# Example usage
if __name__ == "__main__":
    """ 
    Args:
        input_path: Complete path to input.json file for Taichi MPM  
        material_feature: Whether to add material properties to node feature (Boolean)  
    """
    ## Parse through command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="temp_mpm_input.json", type=str, help="Input json file name")
    parser.add_argument('--material_feature', default=False, type=bool, help="Whether to add material property to node feature")
    parser.add_argument('--mode', default="train", type=str, help="Mode to output .npz file. Options are: 'train' 'test' or 'valid' .npz file.")
    # parser.add_argument('--output_dir', default="metadataoutput.json", type=str, help="Output directory to place .npz files") # Not needed
    args = parser.parse_args()
    input_path = args.input_path
    
    # Load the json file
    f = open(input_path)
    inputs = json.load(f)
    output_dir = inputs['save_path'] # Made the save path the output directory
    num_traject = inputs['id_range'][1]
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
    # Initialize instance of data handler using base directory {npz_directory} No '/' at the end (?)
    handler = TrajectoryDataHandler(npz_directory)

    # Inspect a specific .npz file
    # handler.inspect_npz_file("test.npz")

    # Merge trajectories (multiple .npz files) into a single .npz file
    handler.merge_trajectories(f"{args.mode}.npz", num_trajectories=100)
    
    # Inspect the newly merged and created train.npz file
    handler.inspect_npz_file(f"{args.mode}.npz")
    
    # Metadata file for the training dataset
    if args.mode == 'train':
        # Calculate and create metadata.json file
        handler.create_metadata("metadata", num_trajectories=100)
        
    # Notes: Necessary metadata (9):
    # {"bounds","sequence_length", "default_connectivity_radius","dim", "dt", "vel_mean", "vel_std", "acc_mean", "acc_std"}
    # Note: In GNS train.py sequence_length not necessary but recommended and dt is not necessary, only describing MPM simulation timesteps.