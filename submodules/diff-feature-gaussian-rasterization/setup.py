############################################################
# Code for FiT3D 
# by Yuanwen Yue
# Stage 1: Lifting 2D features to feature Gaussians
############################################################
# Code was modified from Gaussian Splatting codebase
# https://github.com/graphdeco-inria/gaussian-splatting
# Copyright (C) 2023, Inria, GRAPHDECO research group
############################################################

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_feature_gaussian_rasterization",
    packages=['diff_feature_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_feature_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
