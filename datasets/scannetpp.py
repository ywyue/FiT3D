############################################################
# Code for FiT3D 
# by Yuanwen Yue
# Stage 2: 3D-aware fine-tuning
############################################################

import albumentations as A
import numpy as np
import os
import torch
import torch.utils.data

from pathlib import Path
from PIL import Image
from scene.cameras import Camera
from torch import nn
from torch.utils.data import Dataset

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class ScanNetPP(Dataset):
    def __init__(self, root, sample_list, gaussian_list, view_list, transforms, sh_degree=3):
        super(ScanNetPP, self).__init__()

        self.root = root
        self.transforms = transforms
        self.ids = np.loadtxt(sample_list, dtype=str)
    
        self.view_dict = np.load(view_list, allow_pickle=True).tolist()
        self.sh_degree = sh_degree
    
        pretrained_gaussians = torch.load(gaussian_list)
        self.gaussians_model_param = {}
        self.gaussians_feat_cnn = {}
        
        for scene_id, scene_data in pretrained_gaussians.items():
            self.gaussians_model_param[scene_id] = {key:value for key, value in scene_data.items() if key != 'feat_cnn'}
            high_sem_dim, low_sem_dim = scene_data['feat_cnn']['0.weight'].shape[:2]
            feat_cnn = nn.Sequential(
                nn.Conv2d(low_sem_dim, high_sem_dim, kernel_size=3, padding=1, bias=False),
                )
            feat_cnn.load_state_dict(scene_data['feat_cnn'])
            self.gaussians_feat_cnn[scene_id] = feat_cnn

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        img_id = self.ids[index]
        scene_name, img_name = img_id.split('_')

        original_image = np.array(Image.open(os.path.join(self.root, scene_name, 'images', img_name+'.JPG')))

        ### Specifies the pixels that have been anonymized, will be masked out in the loss
        anno_map = np.array(Image.open(os.path.join(self.root, scene_name, 'masks', img_name+'.png')))
        anno_map = np.expand_dims(anno_map, axis=-1)

        ### get image
        if self.transforms is not None:
            transformed = self.transforms(image=original_image, mask=anno_map)
            image, anno_map = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])
        else:
            image, anno_map = torch.tensor(original_image), torch.LongTensor(anno_map)

        image = image.permute(2, 0, 1)
        anno_map = anno_map[:, :, 0:1].permute(2, 0, 1) / 255.0

        ### get camera pose
        view = self.view_dict[img_id]

        ### get pre-trained Gaussian
        gaussian_model_param = self.gaussians_model_param[scene_name]
        gaussian_feat_cnn = self.gaussians_feat_cnn[scene_name]

        return scene_name, image, anno_map, view, img_name, gaussian_model_param, gaussian_feat_cnn



def make_transforms(image_set):

    if image_set == 'train':
        return A.Compose([
            A.Normalize(mean=list(ADE_MEAN), std=list(ADE_STD)),
    ])
        
    if image_set == 'val' or image_set == 'test':
        return A.Compose([
            A.Normalize(mean=list(ADE_MEAN), std=list(ADE_STD)),
    ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f'provided data path {root} does not exist'

    Sample_PATHS = {
        "train": args.train_list,
        "val": args.val_list,
    }

    Gaussian_PATHS = {
        "train": args.train_gaussian_list,
        "val": args.val_gaussian_list,
    }

    View_PATHS = {
        "train": args.train_view_list,
        "val": args.val_view_list,
    }

    sample_list = Sample_PATHS[image_set]
    gaussian_list = Gaussian_PATHS[image_set]
    view_list = View_PATHS[image_set]
    
    dataset = ScanNetPP(root, sample_list, gaussian_list, view_list, transforms=make_transforms(image_set))
    
    return dataset
