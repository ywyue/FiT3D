import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm



def main():


    train_samples_list = []
    val_samples_list = []

    scene_root = '../db/scannetpp/scenes'
    metadata_folder = '../db/scannetpp/metadata'

    train_scenes = np.loadtxt(os.path.join(metadata_folder, 'nvs_sem_train.txt'), dtype=str)
    val_scenes = np.loadtxt(os.path.join(metadata_folder, 'nvs_sem_val.txt'), dtype=str)

    for scene in train_scenes:
        scene_imgs_folder = os.path.join(scene_root, scene, "images")
        image_ids = os.listdir(scene_imgs_folder)
        sample_ids = [scene+ '_' + image_id.split('.JPG')[0] for image_id in image_ids]
        train_samples_list += sample_ids

    for scene in val_scenes:
        scene_imgs_folder = os.path.join(scene_root, scene, "images")
        image_ids = os.listdir(scene_imgs_folder)
        sample_ids = [scene+ '_' + image_id.split('.JPG')[0] for image_id in image_ids]
        val_samples_list += sample_ids


    np.savetxt(os.path.join(metadata_folder, 'train_samples.txt'), np.array(train_samples_list), fmt='%s')
    np.savetxt(os.path.join(metadata_folder, 'val_samples.txt'), np.array(val_samples_list), fmt='%s')
    
    

if __name__ == "__main__":
    

    main()
    

