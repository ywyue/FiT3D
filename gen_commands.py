import argparse
import os
from tqdm import tqdm
import numpy as np


def config():
    a = argparse.ArgumentParser(description='Generate commands for training feature Gaussians for ScanNet++')
    a.add_argument('--train_fgs_commands_folder', default='train_fgs_commands', type=str, \
                  help='folder to store commands for training feature Gaussians')
    a.add_argument('--model_name', default='dinov2_small', type=str, \
                  help='2D feature extractor name, select from dinov2_small, dinov2_base, dinov2_reg_small, clip_base \
                       mae_base, deit3_base')
    a.add_argument('--low_sem_dim', default=64, type=str, \
                  help='low semantic feature dimension for each Gaussian')
    args = a.parse_args()
    return args

def main(args):

    train_scenes = np.loadtxt('db/scannetpp/metadata/nvs_sem_train.txt', dtype=str)
    val_scenes = np.loadtxt('db/scannetpp/metadata/nvs_sem_val.txt', dtype=str)

    all_scenes = list(train_scenes) + list(val_scenes)

    train_fgs_commands_folder = args.train_fgs_commands_folder
  
    os.makedirs(train_fgs_commands_folder, exist_ok=True)

    model_name = args.model_name
    low_sem_dim = args.low_sem_dim

    for idx, scene in enumerate(tqdm(all_scenes)):

        scene_id = f'{idx:03}'

        with open (os.path.join(train_fgs_commands_folder,'{}_{}.sh'.format(scene_id, scene)), 'w') as rsh:
            rsh.write('''#! /bin/bash
ulimit -n 4096
conda activate fit3d
python train_feat_gaussian.py --run_name=scene_{}_{}_{} \\
                    --model_name={} \\
                    --source_path=db/scannetpp/scenes/{} \\
                    --low_sem_dim={}
                '''.format(scene_id, scene, model_name, model_name, scene, low_sem_dim))
            
    print('Done')

if __name__ == "__main__":

    main(config())
    