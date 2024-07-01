import os
import sys
import numpy as np
import trimesh
import torch

sys.path.append("..")
from smpl_pytorch.body_models import SMPL
from utils_drape import draping, load_udf, load_lbs, reconstruct

def load_pose_and_beta_from_obj(obj_file):
    # Placeholder function to extract pose and beta from the obj file
    # This function needs to be implemented correctly based on how the pose and beta are stored in the obj file
    # Here we assume the obj file has pose and beta stored in some manner
    mesh = trimesh.load(obj_file)
    # Extract pose and beta from the mesh or obj file
    pose = torch.zeros(1, 72)  # Example pose with 72 parameters
    beta = torch.zeros(1, 10)
    
        
    left_shoulder_idx = 16 * 3  # 16th joint, each joint has 3 values
    right_shoulder_idx = 17 * 3
    left_hip_idx = 1 * 3
    right_hip_idx = 2 * 3
    
    pose[0, left_shoulder_idx + 1] = np.deg2rad(-10)  # Move downward
    pose[0, right_shoulder_idx + 1] = np.deg2rad(10)  # Move downward
    pose[0, left_shoulder_idx + 2] = np.deg2rad(-60)  # Rotate outward
    pose[0, right_shoulder_idx + 2] = np.deg2rad(60)
    # Spread legs outward
    pose[0, left_hip_idx + 2] = np.deg2rad(10)  # Rotate outward
    pose[0, right_hip_idx + 2] = np.deg2rad(-10)# Placeholder: replace with actual extraction logic
    return pose, beta

def main(pose, beta, checkpoints_dir, extra_dir, smpl_model_dir, output_folder, device, top_idx=0, bottom_idx=0, resolution=256):
    ''' Load pretrained models '''
    models = load_lbs(checkpoints_dir, device)
    _, latent_codes_top, decoder_top = load_udf(checkpoints_dir, 'top_codes.pt', 'top_udf.pt', device)
    coords_encoder, latent_codes_bottom, decoder_bottom = load_udf(checkpoints_dir, 'bottom_codes.pt', 'bottom_udf.pt', device)

    ''' Initialize SMPL model '''
    data_body = np.load(os.path.join(extra_dir, 'avatar_info.npz'))
    tfs_c_inv = torch.FloatTensor(data_body['tfs_c_inv']).to(device)
    smpl_server = SMPL(model_path=smpl_model_dir, gender='m').to(device)

    ''' Reconstruct shirt(top_idx)/pants(bottom_idx) in T-pose '''
    mesh_top, vertices_top_T, faces_top = reconstruct(coords_encoder, decoder_top, latent_codes_top[[top_idx]], udf_max_dist=0.1, resolution=resolution, differentiable=False)
    mesh_top.export(output_folder + '/top-T.obj')
    mesh_bottom, vertices_bottom_T, faces_bottom = reconstruct(coords_encoder, decoder_bottom, latent_codes_bottom[[bottom_idx]], udf_max_dist=0.1, resolution=resolution, differentiable=False)
    mesh_bottom.export(output_folder + '/bottom-T.obj')

    ''' Skinning garments '''
    vertices_Ts = [vertices_top_T, vertices_bottom_T]
    faces_garments = [faces_top.cpu().numpy(), faces_bottom.cpu().numpy()]
    latent_codes = [latent_codes_top[[top_idx]], latent_codes_bottom[[bottom_idx]]]

    # Move pose and beta to the same device
    pose = pose.to(device)
    beta = beta.to(device)

    top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh = draping(vertices_Ts, faces_garments, latent_codes, pose, beta, models, smpl_server, tfs_c_inv)
    body_mesh.export(output_folder + '/body.obj')
    top_mesh.export(output_folder + '/shirt.obj')
    bottom_mesh.export(output_folder + '/pants.obj')
    bottom_mesh_layer.export(output_folder + '/pants-layer.obj')

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    ''' dir to load models and extra data '''
    checkpoints_dir = '../checkpoints'
    extra_dir = '../extra-data'
    smpl_model_dir = '../smpl_pytorch'

    ''' dir to dump mesh '''
    output_folder = '../output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ''' Extract pose and beta from result.obj '''
    pose, beta = load_pose_and_beta_from_obj('/home/hello/drapenet/result.obj')  # Ensure the path to result.obj is correct

    ''' top_idx/bottom_idx - the index corresponding to the shirt/pants '''
    main(pose, beta, checkpoints_dir, extra_dir, smpl_model_dir, output_folder, device, top_idx=208, bottom_idx=15, resolution=256)
