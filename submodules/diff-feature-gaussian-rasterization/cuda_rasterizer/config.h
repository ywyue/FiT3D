/*
 * Code for FiT3D: https://github.com/ywyue/FiT3D
 * Modified from https://github.com/graphdeco-inria/gaussian-splatting
 */

/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_CHANNELS_FEAT 64 // Feature dimension for each Gaussian, need to be consistent with low_sem_dim in train_feature_gaussian.py
#define BLOCK_X 16
#define BLOCK_Y 16

#endif