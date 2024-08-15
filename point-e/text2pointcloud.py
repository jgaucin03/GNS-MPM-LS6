import torch
import numpy as np
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from plyfile import PlyData, PlyElement

def save_point_cloud_as_npz(point_cloud, filename):
    coords = point_cloud.coords
    if 'R' in point_cloud.channels:
        colors = np.stack([point_cloud.channels['R'], point_cloud.channels['G'], point_cloud.channels['B']], axis=-1)
        points = np.hstack((coords, colors))
        
    else:
        points = coords
   
    np.savez(filename, points=points)

def save_point_cloud_positions_as_npz(point_cloud, filename):
    positions = point_cloud.coords
    np.savez(filename, points=positions)

def save_point_cloud_as_obj(point_cloud, filename):
    coords = point_cloud.coords
    with open(filename + '.obj', 'w') as f:
        for i in range(coords.shape[0]):
            f.write(f'v {coords[i, 0]} {coords[i, 1]} {coords[i, 2]}\n')
            if 'R' in point_cloud.channels:
                color = (point_cloud.channels['R'][i], point_cloud.channels['G'][i], point_cloud.channels['B'][i])
                f.write(f'vn {color[0]} {color[1]} {color[2]}\n')

def save_point_cloud_as_ply(point_cloud, filename):
    coords = point_cloud.coords
    if 'R' in point_cloud.channels:
        colors = np.stack([point_cloud.channels['R'], point_cloud.channels['G'], point_cloud.channels['B']], axis=-1)
        colors = (colors * 255).astype(np.uint8)  # Convert colors to uint8
        print("Colors shape:", colors.shape)
        print("Sample colors:", colors[:5])  # Print the first 5 colors for debugging
        vertices = np.hstack((coords, colors))
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    else:
        vertices = coords
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = np.array([tuple(vertex) for vertex in vertices], dtype=dtype)
    print("Vertices shape:", vertices.shape)
    print("Sample vertices:", vertices[:5])  # Print the first 5 vertices for debugging
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply.write(filename + '.ply')

def main(prompt, output_filename, output_format='ply'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('Creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('Downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('Downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
    )

    print(f'Setting prompt: {prompt}')
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]

    if output_format == 'npz':
        save_point_cloud_as_npz(pc, f"{output_filename}_og.npz")
        save_point_cloud_positions_as_npz(pc, f"{output_filename}_pos.npz")
    elif output_format == 'obj':
        save_point_cloud_as_obj(pc, output_filename)
    elif output_format == 'ply':
        save_point_cloud_as_ply(pc, output_filename)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f'Point cloud saved as {output_filename}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a point cloud from a text prompt using point-e.')
    parser.add_argument('--prompt', type=str, required=True, help='The text prompt to generate the point cloud.')
    parser.add_argument('--output_filename', type=str, required=True, help='The output file to save the point cloud.')
    parser.add_argument('--format', type=str, choices=['npz', 'obj', 'ply'], default='ply', help='The output file format (default: ply).')

    args = parser.parse_args()
    main(args.prompt, args.output_filename, args.format)