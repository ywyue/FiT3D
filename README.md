# Improving 2D Feature Representations by 3D-Aware Fine-Tuning
### ECCV 2024

[Yuanwen Yue](https://ywyue.github.io/) <sup>1</sup>,
[Anurag Das](https://anurag-198.github.io/) <sup>2</sup>,
[Francis Engelmann](https://francisengelmann.github.io/) <sup>1,3</sup>,
[Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html) <sup>1</sup>,
[Jan Eric Lenssen](https://janericlenssen.github.io/) <sup>2</sup>
<br>

**<sup>1</sup>ETH Zurich, <sup>2</sup>Max Planck Institute for Informatics, <sup>3</sup>Google**

### [Project Page](https://ywyue.github.io/FiT3D) | [Paper](http://arxiv.org/abs/2407.20229) 

<a target="_blank" href="https://colab.research.google.com/github/ywyue/FiT3D/blob/main/FiT3D_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://huggingface.co/spaces/yuanwenyue/FiT3D">
  <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>

<img width="1100" src="./assets/teaser.png" />

This is the official repository for the paper Improving 2D Feature Representations by 3D-Aware Fine-Tuning.

## Changelog
- [x] Add Colab Notebook and Hugging Face demo
- [x] Release ScanNet++ preprocessing code
- [x] Release feature Gaussian training code
- [x] Release fine-tuning code
- [x] Release evaluation code

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#preparation">Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Demo
We provide a [Colab Notebook](https://colab.research.google.com/github/ywyue/FiT3D/blob/main/FiT3D_demo.ipynb) with step-by-step guides to make inference and visualize the PCA features and K-Means clustering of original 2D models and our fine-tuned models.
We also provide an online [Hugging Face demo 🤗](https://huggingface.co/spaces/yuanwenyue/FiT3D) where users can upload their own images and check the visualizations online. Alternatively, to run the demo locally, just try ```python app.py```.

<img width="1100" src="./assets/demo.gif" />



## Preparation

### Environment
* The code has been tested on Linux with Python 3.10.14, torch 1.9.0, and cuda 11.8.
* Create an environment and install pytorch and other required packages:
  ```shell
  git clone https://github.com/ywyue/FiT3D.git
  cd FiT3D
  conda create -n fit3d python=3.10
  conda activate fit3d
  pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
* Compile the feature rasterization modules and the knn module for feature lifting:
  ```shell
  cd submodules/diff-feature-gaussian-rasterization
  python setup.py install
  cd ../simple-knn/
  python setup.py install
  ```
* Install ```mmcv``` and ```mmsegmentation```, required for downstream evaluation. Note we modifed the source code so please build them from source as follows:
  ```shell
  cd mmcv
  MMCV_WITH_OPS=1 pip install -e . -v
  cd ../mmsegmentation
  pip install -e . -v
  ```

### Data
We train feature Gaussians and fine-tuning on ScanNet++ scenes. Preprocessing code and instructions are [here](https://github.com/ywyue/FiT3D/tree/main/scannetpp_preprocess). After preprocessing, the ScanNet++ data is expected to be organized as following:
```
FiT3D/
└── db/
    └── scannetpp/
        ├── metadata/
        |    ├── nvs_sem_train.txt  # Training set for NVS and semantic tasks with 230 scenes
        |    ├── nvs_sem_val.txt # Validation set for NVS and semantic tasks with 50 scenes
        |    ├── train_samples.txt  # Training sample list, formatted as sceneID_imageID
        |    ├── val_samples.txt # Validation sample list, formatted as sceneID_imageID
        |    ├── train_view_info.npy  # Training sample camera info, e.g. projection matrices
        |    └── val_view_info.npy # Validation sample camera info, e.g. projection matrices
        └── scenes/
            ├── 0a5c013435  # scene id
            ├── ...
            └── 0a7cc12c0e
              ├── images  # undistorted and downscaled images
              ├── masks # undistorted and downscaled anonymized masks
              ├── points3D.txt  # 3D feature points used by COLMAP
              └── transforms_train.json # camera poses in the format used by Nerfstudio
```

For all other evaluation datasets (ScanNet, NYUd, NYUv2, ADE20k, Pascal VOC, KITTI), please follow their official websites for downloading instructions.

## Training
### Stage I: Lifting Features to 3D

Example command to train the feature Gaussians for a single scene: 
```shell
python train_feat_gaussian.py --run_name=example_feature_gaussian_training \
                    --model_name=dinov2_small \
                    --source_path=db/scannetpp/scenes/0a5c013435 \
                    --low_sem_dim=64
```
```model_name``` indicates the 2D feature extractor and can be selected from ```dinov2_small```, ```dinov2_base```, ```dinov2_reg_small```, ```clip_base```, ```mae_base```, ```deit3_base```. ```low_sem_dim``` is the dimension of the semantic feature vector attached to each Gaussian. Note it should have the same value with ```NUM_CHANNELS_FEAT``` in ```submodules/diff-feature-gaussian-rasterization/cuda_rasterizer/config.h```.

To generate the commands for training Gaussians for all scenes in ScanNet++, run:
```shell
python gen_commands.py --train_fgs_commands_folder=train_fgs_commands --model_name=dinov2_small --low_sem_dim=64
```
Training commands for all scenes will be stored in ```train_fgs_commands```.

After training, we need to write the parameters of all feature Gaussians to a single file, which will be used in the 2nd stage. To do that, run:
```shell
python write_feat_gaussian.py
```
After that, all the pretrained Gaussians of training scenes are stored as ```pretrained_feat_gaussians_train.pth``` and all the pretrained Gaussians of validation scenes are stored as ```pretrained_feat_gaussians_val.pth```. Both files will be stored in ```db/scannetpp/metadata```.

### Stage II: Fine-Tuning

In this stage, we use the pretrained Gaussians to render features and use those features as target to finetune the 2D feature extractor. To do that, run
```shell
python finetune.py --model_name=dinov2_small \
                   --output_dir=output_finemodel \
                   --job_name=finetuning_dinov2_small \
                   --train_gaussian_list=db/scannetpp/metadata/pretrained_feat_gaussians_train.pth \
                   --val_gaussian_list=db/scannetpp/metadata/pretrained_feat_gaussians_val.pth

```
```model_name``` indicates the 2D feature extractor and should be consistent with the feature extractor used in the first stage. The default fine-tuning epoch is 1, after which the weights of the finetuned model will be saved in ```output_dir/date_job_name```.

## Evaluation

For visual comparison of PCA features and K-means clustering results, please check our [Colab Notebook](https://colab.research.google.com/github/ywyue/FiT3D/blob/main/FiT3D_demo.ipynb) and demo ```app.py```.

For quantitative evaluation on downstream tasks, we conduct linear probing evaluation on semantic segmentation and depth estimation.

* First, download all evaluation dataset and put them in ```eval_data``` as follows:
  ```
  eval_data/
  ├── scannetpp # semantic segmentation and depth estimation
  ├── scannet # semantic segmentation and depth estimation
  ├── nyu # depth estimation
  ├── nyuv2 # semantic segmentation
  ├── kitti # depth estimation
  ├── ADEChallengeData2016 # semantic segmentation
  ├── VOC2012 # semantic segmentation
  └── kitti # depth estimation
  ```

Processed scannetpp dataset with annotations can be downloaded from [here](https://drive.google.com/file/d/18BGnCzk51nv79M-SiJ6ezX2WOTcQDZyi/view?usp=sharing). For other datasets, please download from their official websites and their annotations are already provided so no (or only little) preprocessing needed.

* Launch the linear probing evaluation. We provide two example scripts: ```eval_scripts/fit3d/linear_eval_sem.sh``` for semantic segmentation and ```eval_scripts/fit3d/linear_eval_depth.sh``` for depth estimation. Note:
  * Change  ```model``` and ```dataset``` to adapt the script for evaluation with different models and datasets. See comments in the script.
  * The default linear probing evaluation requires **8 GPUs** for **40K** iterations for semantic segmentation and **38400** iterations for depth estimation. If you use fewer GPUs then the number of iterations needs to be linearly increased. For example, if ```ngpu``` is set as **4**, then **80K** iterations are required for semantic segmentation (76800 iterations for depth estimation). The number of iterations (parameter called ```max_iters```) can be modified from respective ```config``` files.

* To evaluate baseline models (i.e. original 2D models), see ```eval_scripts/baseline/linear_eval_sem.sh``` and ```eval_scripts/baseline/linear_eval_depth.sh```

## Citation

If you find our code or paper useful, please cite:
```
@inproceedings{yue2024improving,
  title     = {{Improving 2D Feature Representations by 3D-Aware Fine-Tuning}},
  author    = {Yue, Yuanwen and Das, Anurag and Engelmann, Francis and Tang, Siyu and Lenssen, Jan Eric},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
