import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm
from gaussian_renderer import GaussianModel



def write_gaussians(gaussians_dict):
    gaussian_models = {}
    for scene_name, gaussian_path in tqdm(gaussians_dict.items()):

        print('loading ======> ' + scene_name)
        model_params = torch.load(os.path.join(gaussian_path, 'chkpnt30000.pth'), 'cpu')
        high_sem_dim, low_sem_dim = model_params['0.weight'].shape[:2]

        gaussians = GaussianModel(3, low_sem_dim, high_sem_dim)

        gaussians.load_ply(os.path.join(gaussian_path,"point_cloud", "iteration_30000","point_cloud.ply"))
        
        means3D = gaussians.get_xyz
        shs = gaussians.get_features
        sem = gaussians.get_sem
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        gaussian_models[scene_name] = {
            'feat_cnn': model_params,
            'means3D': means3D,
            'shs': shs,
            'sem': sem,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations
        }

    return gaussian_models

def config():
    a = argparse.ArgumentParser(description='Write pretrained feature Gaussians to .pth')
    a.add_argument('--pretrained_feat_gaussians_train', default='db/scannetpp/metadata/pretrained_feat_gaussians_train.pth', type=str, \
                  help='path to store pretrained feature Gaussians of all training scenes')
    a.add_argument('--pretrained_feat_gaussians_val', default='db/scannetpp/metadata/pretrained_feat_gaussians_val.pth', type=str, \
                  help='path to store pretrained feature Gaussians of all validation scenes')
    args = a.parse_args()
    return args

def main(args):

    train_scenes = np.loadtxt('db/scannetpp/metadata/nvs_sem_train.txt', dtype=str)
    val_scenes = np.loadtxt('db/scannetpp/metadata/nvs_sem_val.txt', dtype=str)

    gaussian_dict_train = {}
    gaussian_dict_val = {}

    output_folder = 'output' 
    outputs = os.listdir(output_folder)
    for output in tqdm(outputs):
        scene_name = output.split('_')[2]

        if os.path.isfile(os.path.join(output_folder, output, 'chkpnt30000.pth')) and os.path.isfile(os.path.join(output_folder, output, 'point_cloud', 'iteration_30000','point_cloud.ply')):
            if scene_name in train_scenes:
                gaussian_dict_train[scene_name] = os.path.join(output_folder, output)
            
            elif scene_name in val_scenes:
                gaussian_dict_val[scene_name] = os.path.join(output_folder, output)

    assert len(gaussian_dict_train) == 230
    assert len(gaussian_dict_val) == 50

    train_gaussian_models = write_gaussians(gaussian_dict_train)
    torch.save(train_gaussian_models, args.pretrained_feat_gaussians_train)
    print("done for training scenes")

    val_gaussian_models = write_gaussians(gaussian_dict_val)
    torch.save(val_gaussian_models, args.pretrained_feat_gaussians_val)
    print("done for validation scenes")
    

if __name__ == "__main__":

    main(config())
    


