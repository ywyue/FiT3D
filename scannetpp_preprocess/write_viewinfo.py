import numpy as np
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.getcwd()))
from scene.dataset_readers import readCamerasFromTransforms
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_cameras(scene_root, scenes, file_path):

    cameras = {}

    for scene in tqdm(scenes):
        cam_infos = readCamerasFromTransforms(os.path.join(scene_root, scene), "transforms_train.json", False, ".png")
    
        for cam_info in cam_infos:

            world_view_transform = torch.tensor(getWorld2View2(cam_info.R, cam_info.T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).to(device)
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=cam_info.FovX, fovY=cam_info.FovY).transpose(0,1).to(device)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            sample_id = scene + '_' + cam_info.image_name
            cameras[sample_id] = {
                'FoVx': cam_info.FovX,
                'FoVy': cam_info.FovY,
                'image_width': cam_info.image.size[0],
                'image_height': cam_info.image.size[1],
                'world_view_transform': world_view_transform.cpu().numpy(),
                'full_proj_transform': full_proj_transform.cpu().numpy(), 
                'camera_center': camera_center.cpu().numpy()
            }
            
    np.save(file_path, cameras)


def main():

    train_cameras = {}
    val_cameras = {}

    scene_root = '../db/scannetpp/scenes'
    metadata_folder = '../db/scannetpp/metadata'

    train_scenes = np.loadtxt(os.path.join(metadata_folder, 'nvs_sem_train.txt'), dtype=str)
    val_scenes = np.loadtxt(os.path.join(metadata_folder, 'nvs_sem_val.txt'), dtype=str)
    write_cameras(scene_root, train_scenes, os.path.join(metadata_folder, 'train_view_info.npy'))
    write_cameras(scene_root, val_scenes,  os.path.join(metadata_folder, 'val_view_info.npy'))


if __name__ == "__main__":
    

    main()
    