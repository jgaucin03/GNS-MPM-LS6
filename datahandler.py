import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class TrajectoryDataHandler:
    def __init__(self, npz_directory):
        self.npz_directory = npz_directory

    def load_npz(self, file_path):
        return np.load(file_path, allow_pickle=True)

    def inspect_trajectory(self, data, key):
        trajectory_data = data[key]
        positions = trajectory_data[0]
        particle_types = trajectory_data[1]

        # Optional: Check if material_properties are present
        if len(trajectory_data) > 2:
            material_properties = trajectory_data[2]
            print(f"Material properties shape: {material_properties.shape}")

        # Extracting the shape information
        n_timesteps, n_particles, n_dims = positions.shape

        print(f"Trajectory: {key}")
        print(f"Positions shape: {positions.shape}")
        print(f"Number of time steps: {n_timesteps}")
        print(f"Number of particles: {n_particles}")
        print(f"Number of dimensions: {n_dims}")
        print(f"Particle types shape: {particle_types.shape}")
        print(f"First few particle types: {particle_types[:10]}")
        print(f"First particle positions: {positions[:1]}")
        print()

        return n_timesteps * n_particles, n_particles

    def inspect_npz_file(self, file_name):
        file_path = os.path.join(self.npz_directory, file_name)
        data = self.load_npz(file_path)

        total_samples = 0
        num_particles_list = []

        for key in data.files:
            num_samples, num_particles = self.inspect_trajectory(data, key)
            total_samples += num_samples
            num_particles_list.append(num_particles)

        print(f"Total samples: {total_samples}")

        # Plotting the number of particles for each trajectory
        """ plt.figure(figsize=(10, 6))
        plt.plot(range(len(num_particles_list)), num_particles_list, marker='o', linestyle='-', color='b')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Number of Particles')
        plt.title('Number of Particles in Each Trajectory')
        plt.grid(True)
        plt.show() """

    def merge_trajectories(self, output_file_name, num_trajectories=10):
        combined_trajectories = {}

        for i in range(num_trajectories):
            file_path = os.path.join(self.npz_directory, f"trajectory{i}.npz")
            with self.load_npz(file_path) as data:
                key = f"trajectory{i}"
                positions, particle_types = data[key]
                combined_trajectories[f"simulation_trajectory_{i}"] = (positions, particle_types)

        output_file = os.path.join(self.npz_directory, output_file_name)
        print("Compressing npz...")
        np.savez_compressed(output_file, **combined_trajectories)
        print(f"Combined trajectories saved to {output_file}")
        
    # Calculate metadata for a merged npz file:
    
    def create_metadata(self, output_file_name, num_trajectories=10, sequence_length=350, default_connectivity_radius=0.025, dim=3, material_feature_len=0, dt_mpm=0.0025, dt_gns=1.0):
        bounds = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
        mpm_cell_size = [None, None, None]
        nparticles_per_cell = None

        trajectories = {}
        
        # data_names = []
        cumulative_count = 0
        cumulative_sum_vel = np.zeros((1, dim))
        cumulative_sum_acc = np.zeros((1, dim))
        cumulative_sumsq_vel = np.zeros((1, dim))
        cumulative_sumsq_acc = np.zeros((1, dim))

        for i in tqdm(range(num_trajectories)):
            data_name = f"trajectory{i}"
            # Not necessary to keep track of this for metadata
            # data_names.append(data_name)
            npz_path = os.path.join(self.npz_directory, f"{data_name}.npz")
            data = self.load_npz(npz_path)
            for simulation_id, trajectory in data.items():
                trajectories[f"simulation_trajectory_{i}"] = (trajectory)

            positions = trajectory[0]
            array_shape = positions.shape
            flattened_positions = np.reshape(positions, (-1, array_shape[-1]))

            velocities = np.empty_like(positions)
            velocities[1:] = (positions[1:] - positions[:-1]) / dt_gns
            velocities[0] = 0
            flattened_velocities = np.reshape(velocities, (-1, array_shape[-1]))

            accelerations = np.empty_like(velocities)
            accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt_gns
            accelerations[0] = 0
            flattened_accelerations = np.reshape(accelerations, (-1, array_shape[-1]))

            cumulative_count += len(flattened_velocities)
            cumulative_sum_vel += np.sum(flattened_velocities, axis=0)
            cumulative_sum_acc += np.sum(flattened_accelerations, axis=0)
            cumulative_sumsq_vel += np.sum(flattened_velocities**2, axis=0)
            cumulative_sumsq_acc += np.sum(flattened_accelerations**2, axis=0)
            cumulative_mean_vel = cumulative_sum_vel / cumulative_count
            cumulative_mean_acc = cumulative_sum_acc / cumulative_count
            cumulative_std_vel = np.sqrt(
                (cumulative_sumsq_vel / cumulative_count - (cumulative_sum_vel / cumulative_count)**2))
            cumulative_std_acc = np.sqrt(
                (cumulative_sumsq_acc / cumulative_count - (cumulative_sum_acc / cumulative_count)**2))

        statistics = {
            "mean_velocity_x": float(cumulative_mean_vel[:, 0]),
            "mean_velocity_y": float(cumulative_mean_vel[:, 1]),
            "mean_velocity_z": float(cumulative_mean_vel[:, 2]),
            "std_velocity_x": float(cumulative_std_vel[:, 0]),
            "std_velocity_y": float(cumulative_std_vel[:, 1]),
            "std_velocity_z": float(cumulative_std_vel[:, 2]),
            "mean_accel_x": float(cumulative_mean_acc[:, 0]),
            "mean_accel_y": float(cumulative_mean_acc[:, 1]),
            "mean_accel_z": float(cumulative_mean_acc[:, 2]),
            "std_accel_x": float(cumulative_std_acc[:, 0]),
            "std_accel_y": float(cumulative_std_acc[:, 1]),
            "std_accel_z": float(cumulative_std_acc[:, 2])
        }

        for key, value in statistics.items():
            print(f"{key}: {value:.7E}")
        ## Note: In GNS train.py: 
        # Default Boundary Augment (1.0), no 'Default' connectivity radius (0.025), default sequence_length default is found from positions.shape[1]
        # Default nnode_in is 37 if 3D else 30, default nedge_in is 1 + dimensions (e. g. 3 + 1 = 4 nedge_in)
        ## Not used: material_feature_len, dt, mpm_cell_size, nparticles_per_cell, data_names
        metadata = {
            "bounds": bounds,
            "sequence_length": sequence_length,
            "default_connectivity_radius": default_connectivity_radius,
            "boundary_augment": 1.0,
            "material_feature_len": material_feature_len,
            "dim": dim,
            "dt": dt_mpm,
            "vel_mean": [statistics["mean_velocity_x"], statistics["mean_velocity_y"], statistics["mean_velocity_z"]],
            "vel_std": [statistics["std_velocity_x"], statistics["std_velocity_y"], statistics["std_velocity_z"]],
            "acc_mean": [statistics["mean_accel_x"], statistics["mean_accel_y"], statistics["mean_accel_z"]],
            "acc_std": [statistics["std_accel_x"], statistics["std_accel_y"], statistics["std_accel_z"]],
            "mpm_cell_size": mpm_cell_size,
            "nparticles_per_cell": nparticles_per_cell,
        }

        metadata_file_path = os.path.join(self.npz_directory, f"{output_file_name}.json")
        # Open the metadata json file, make it if it doesn't exist
        with open(metadata_file_path, "w") as fp:
            json.dump(metadata, fp)
        print(f"metadata saved at: {metadata_file_path}")

# Example usage
if __name__ == "__main__":
    npz_directory = "/scratch/10029/jgaucin/taichi_mpm_water/saved"
    # Initialize instance of data handler using base directory {npz_directory}
    handler = TrajectoryDataHandler(npz_directory)

    # Inspect a specific .npz file
    handler.inspect_npz_file("train.npz")

    # Merge trajectories into a single .npz file
    # handler.merge_trajectories("train.npz", num_trajectories=10)

    # Calculate and save metadata
    handler.calc_metadata("metadata", num_trajectories=100)
    # Necessary metadata (9):
    # {"bounds","sequence_length", "default_connectivity_radius","dim", "dt", "vel_mean", "vel_std", "acc_mean", "acc_std"}
    # Note: In GNS train.py sequence_length not necessary but recommended and dt is not necessary, only describing MPM simulation timesteps.