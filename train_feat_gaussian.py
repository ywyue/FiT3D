############################################################
# Code for FiT3D 
# by Yuanwen Yue
# Stage 1: Lifting 2D features to feature Gaussians
############################################################
# Code was modified from Gaussian Splatting codebase
# https://github.com/graphdeco-inria/gaussian-splatting
# Copyright (C) 2023, Inria, GRAPHDECO research group
############################################################

import datetime
import os
import sys
import torch
import numpy as np
import uuid
import wandb

from random import randint
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.general_utils import safe_state, setup_wandb
from utils.loss_utils import l1_loss, ssim, l2_loss
from utils.model_utils import forward_2d_model, build_2d_model, viz_feat

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from, low_sem_dim, model_name, run_name):

    device = torch.device('cuda')

    ### load pre-trained 2D feature extractor
    feature_extractor = build_2d_model(model_name=model_name)
    feature_extractor.eval()
    feature_extractor.to(device)

    ### init Gaussian model
    gaussians = GaussianModel(dataset.sh_degree, low_sem_dim, feature_extractor.embed_dim)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.feat_cnn.to(device)

    first_iter = 0

    if checkpoint:
        model_params = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render color and features
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, featmap, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_featmap"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        projected_featmap = gaussians.feat_cnn(featmap.unsqueeze(0)).squeeze(0)
        ### check rendered features
        # viz_feat(projected_featmap.unsqueeze(0), "check_rendered_feat.png")

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image_untouched = viewpoint_cam.original_image_untouched
        height = gt_image.shape[1]
        width = gt_image.shape[2]
        image_name = viewpoint_cam.image_name

        ### forward 2D feature extractor to obtain original features as target
        with torch.no_grad():
            gt_feat_low = forward_2d_model(gt_image_untouched, feature_extractor)

            ### check original features
            # viz_feat(gt_feat_low, "check_original_feat.png")

            gt_featmap = torch.nn.functional.interpolate(gt_feat_low,(height,width), mode ='bilinear').squeeze(0)

        mask = viewpoint_cam.is_masked
        if mask is not None:
            mask = mask.cuda()
            gt_image[mask] = image.detach()[mask]
            feat_mask = mask[:1].expand(*projected_featmap.shape)
            gt_featmap[feat_mask] = projected_featmap.detach()[feat_mask]


        Ll1_feat = l1_loss(projected_featmap, gt_featmap)
        Ll1_color = l1_loss(image, gt_image)
        LSSIM = 1.0 - ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1_color + opt.lambda_dssim * LSSIM + Ll1_feat

        loss.backward()

        iter_end.record()

        # log
        with torch.no_grad():
            train_log_dict = {
                "train/loss_color_l1": Ll1_color,
                "train/loss_color_ssim": LSSIM,
                "train/loss_sem": Ll1_feat,
                "train/loss": loss
                }

            wandb.log(train_log_dict)

            train_log_lr = {}

            for param_group in gaussians.optimizer.param_groups:
                train_log_lr['lr/' + param_group["name"]] = param_group["lr"]

            wandb.log(train_log_lr, step=iteration)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(gaussians.capture(), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Feature Gaussian training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--low_sem_dim", default=64, type=int, help="low semantic feature dimension for each Gaussian. \
                                                                    NOTE: need to change NUM_CHANNELS_FEAT accordingly \
                                                                    in submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h")
    parser.add_argument("--run_name", default='exp', type=str)
    parser.add_argument("--model_name", default='dinov2_small', type=str, help='2D feature extractor. Select from \
                                        dinov2_small, dinov2_base, dinov2_reg_small, clip_base, mae_base, deit3_base')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/",  args.run_name + '_' + unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    print("Optimizing " + args.model_path)


    # setup wandb for logging
    setup_wandb()
    wandb.init(project="Feature_Gaussians")
    wandb.run.name = args.run_name

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.debug_from, args.low_sem_dim, args.model_name, args.run_name
        )

    # All done
    print("\nTraining complete.")
