import argparse
import os
import shutil
from tqdm import tqdm


def config():
    a = argparse.ArgumentParser(description='Reorganize data')
    a.add_argument('--data_root', default='../scannetpp_data', type=str, \
                  help='path to downloaded raw ScanNet++ data')
    args = a.parse_args()
    return args

def main(args):

    data_root = args.data_root

    target_root = "../db/scannetpp/scenes"
    os.makedirs(target_root, exist_ok=True) 

    metadata_root = "../db/scannetpp/metadata"
    os.makedirs(metadata_root, exist_ok=True)
    shutil.move(os.path.join(data_root, "splits", "nvs_sem_train.txt"), metadata_root)
    shutil.move(os.path.join(data_root, "splits", "nvs_sem_val.txt"), metadata_root)


    all_scenes = os.listdir(os.path.join(data_root, "data"))
    for scene in tqdm(all_scenes):
        
        scene_folder = os.path.join(data_root, "data", scene)
        image_folder = os.path.join(scene_folder, "dslr", "undistorted_images")
        mask_folder = os.path.join(scene_folder, "dslr", "undistorted_anon_masks")
        points3d_path = os.path.join(scene_folder, "dslr", "colmap", "points3D.txt")
        transforms_path = os.path.join(scene_folder, "dslr", "nerfstudio", "transforms_undistorted.json")

        target_scene_folder = os.path.join(target_root, scene)
        os.makedirs(target_scene_folder, exist_ok=True) 

        target_image_folder = os.path.join(target_scene_folder, "images")
        target_mask_folder = os.path.join(target_scene_folder, "masks")
        target_points3d_path = os.path.join(target_scene_folder, "points3D.txt")
        target_transforms_path = os.path.join(target_scene_folder, "transforms_train.json")

        shutil.move(image_folder, target_image_folder)
        shutil.move(mask_folder, target_mask_folder)
        shutil.move(points3d_path, target_points3d_path)
        shutil.move(transforms_path, target_transforms_path)
    

if __name__ == "__main__":

    main(config())
    

