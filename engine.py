############################################################
# Code for FiT3D 
# by Yuanwen Yue
# Stage 2: 3D-aware fine-tuning
############################################################

import numpy as np
import torch
import utils.misc as utils
import wandb

from gaussian_renderer import render_fine
from torchvision.transforms.functional import hflip
from utils.model_utils import forward_2d_model_batch


def train_one_epoch(model, criterion,
                    data_loader, optimizer,
                    device, epoch, max_norm):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, batched_inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        target_feats = []
        anno_masks = []
        high_images = []

        with torch.no_grad():
            for batched_input in batched_inputs:
                scene_name = batched_input[0]

                sample_image = batched_input[1].to(device)

                ### move Gaussians to GPU
                gaussians_param = batched_input[5]
                means3D = gaussians_param['means3D'].to(device)
                shs = gaussians_param['shs'].to(device)
                sem = gaussians_param['sem'].to(device)
                opacity = gaussians_param['opacity'].to(device)
                scales = gaussians_param['scales'].to(device)
                rotations = gaussians_param['rotations'].to(device)

                feat_cnn = batched_input[6]
                feat_cnn.to(device)

                ### move cameras to GPU
                view = batched_input[3]
                FoVx = view['FoVx']
                FoVy = view['FoVy']
                image_height = view['image_height']
                image_width = view['image_width']
                world_view_transform = torch.tensor(view['world_view_transform']).to(device)
                full_proj_transform = torch.tensor(view['full_proj_transform']).to(device)
                camera_center = torch.tensor(view['camera_center']).to(device)

                background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

                ### render high-res feature maps from pretrained gaussians as GT feature
                render_pkg = render_fine(FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform,
                                    camera_center, means3D, shs, sem, opacity, scales, rotations, background)
  
                featmap = render_pkg["render_featmap"]

                target_feat = feat_cnn(featmap.unsqueeze(0))

                anno_mask = batched_input[2].to(device)
                
                if np.random.uniform() > 0.5:
                    target_feat = hflip(target_feat)
                    anno_mask = hflip(anno_mask)
                    sample_image = hflip(sample_image)

                target_feats.append(target_feat)
                anno_masks.append(anno_mask.unsqueeze(0))

                high_images.append(sample_image.unsqueeze(0))


        high_images = torch.cat(high_images, dim=0)
        target_feats = torch.cat(target_feats, dim=0)
        anno_masks =  torch.cat(anno_masks, dim=0)

        outputs = forward_2d_model_batch(high_images, model)
        low_h, low_w = outputs.shape[-2:]

        target_feats = torch.nn.functional.interpolate(target_feats, size=(low_h, low_w), mode='bilinear', align_corners=False)
        anno_masks = torch.nn.functional.interpolate(anno_masks, size=(low_h, low_w), mode='bilinear', align_corners=False)

        losses = criterion(outputs, target_feats, anno_masks)

        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # logs
        with torch.no_grad():

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_total_norm)

            if ((i + 1) % 20 == 0) and utils.is_main_process():
                wandb.log({
                    "train/loss": metric_logger.meters['loss'].avg,
                    })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_one_epoch(model, criterion, data_loader_val, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    print_freq = 10

    for i, batched_inputs in enumerate(metric_logger.log_every(data_loader_val, print_freq, header)):

        target_feats = []
        anno_masks = []
        high_images = []

        image_ids = []
        
        with torch.no_grad():
            for batched_input in batched_inputs:
                scene_name = batched_input[0]
                sample_image = batched_input[1].to(device)

                ### move Gaussians to GPU
                gaussians_param = batched_input[5]
                means3D = gaussians_param['means3D'].to(device)
                shs = gaussians_param['shs'].to(device)
                sem = gaussians_param['sem'].to(device)
                opacity = gaussians_param['opacity'].to(device)
                scales = gaussians_param['scales'].to(device)
                rotations = gaussians_param['rotations'].to(device)

                feat_cnn = batched_input[6]
                feat_cnn.to(device)

                ### move cameras to GPU
                view = batched_input[3]
                FoVx = view['FoVx']
                FoVy = view['FoVy']
                image_height = view['image_height']
                image_width = view['image_width']
                world_view_transform = torch.tensor(view['world_view_transform']).to(device)
                full_proj_transform = torch.tensor(view['full_proj_transform']).to(device)
                camera_center = torch.tensor(view['camera_center']).to(device)

                background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

                ### render high-res feature maps from pretrained gaussians as GT feature
                render_pkg = render_fine(FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform,
                                    camera_center, means3D, shs, sem, opacity, scales, rotations, background)
            
                featmap = render_pkg["render_featmap"]

                target_feat = feat_cnn(featmap.unsqueeze(0))

                anno_mask = batched_input[2].to(device)
                
                target_feats.append(target_feat)
                anno_masks.append(anno_mask.unsqueeze(0))
                high_images.append(sample_image.unsqueeze(0))
             

        high_images = torch.cat(high_images, dim=0)
        target_feats = torch.cat(target_feats, dim=0)
        anno_masks =  torch.cat(anno_masks, dim=0)

        outputs = forward_2d_model_batch(high_images, model)

        low_h, low_w = outputs.shape[-2:]

        target_feats = torch.nn.functional.interpolate(target_feats, size=(low_h, low_w), mode='bilinear', align_corners=False)
        anno_masks = torch.nn.functional.interpolate(anno_masks, size=(low_h, low_w), mode='bilinear', align_corners=False)

        losses = criterion(outputs, target_feats, anno_masks)
        loss_value = losses.item()

        metric_logger.update(loss=loss_value)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
