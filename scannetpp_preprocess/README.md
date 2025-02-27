## Data preprocessing for ScanNet++

**Note:** *each new user needs to submit application to request access to ScanNet++ from its official website [official website](https://kaldir.vc.in.tum.de/scannetpp/).* According to the Terms of Use of ScanNet++, we can only share the preprocessed data with people who have also signed the Terms of Use and been granted access to ScanNet++. After you submit your application and get approved from the ScanNet++ team, you can either:

* Forward the approval email to yuayue@ethz.ch and then we will share our preprocessed data with you directly.
* Or follow the data preprocessing instructions as follows to prepare the data yourself.

The code here is largely built on top of [ScanNet++ Toolkit](https://github.com/scannetpp/scannetpp). We downscale the DSLR images by a factor of 2 and undistort them using OpenCV. To prepare data for linear probing evaluation on ScanNet++, we render depth and 2D semantic segmentation from 3D mesh. Note only step 1 and step 2 are required for training feature Gaussians and fine-tuning. 

### Step 1: Download raw dataset

Submit the application and download the data from ScanNet++ [official website](https://kaldir.vc.in.tum.de/scannetpp/). After your application got approved, you will have access to their download script. Please set ```default_assets``` in ```download_scannetpp.yml``` as ```[dslr_train_test_lists_path, dslr_resized_dir, dslr_resized_mask_dir, dslr_colmap_dir, dslr_nerfstudio_transform_path, scan_mesh_path, scan_mesh_mask_path, scan_mesh_segs_path, scan_anno_json_path, scan_sem_mesh_path]```.

### Step 2: Downscale & undistort & write metadata

* Create an environment for data preprocessing and install required packages:
  ```shell
  conda create -n scannetpp python=3.10
  conda activate scannetpp
  pip install -r requirements.txt
  ```
* Downscale DSLR images by a factor of 2. Set ```data_root``` in ```dslr/configs/downscale.yml``` as the folder where the data is downloaded. Then run:
  ```shell
  python -m dslr.downscale dslr/configs/downscale.yml
  ```
* Undistort downscaled images. Set ```data_root``` in ```dslr/configs/downscale.yml``` as above and run:
  ```shell
  python -m dslr.undistort dslr/configs/undistort.yml
  ```
* Reorganize data. Set ```data_root``` as above and run:
  ```shell
  python reorganize_data.py --data_root=../scannetpp_data
  ```
* Generate training and validation samples list used in fine-tuning stage.
  ```shell
  python gen_sample_list.py
  ```
* Pre-compute projection matrices for all training and validation samples and save them to speed up the fine-tuning stage. 
  ```shell
  python write_viewinfo.py
  ```


After this point, the data should be organized as follows correctly:
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
### Step 3: Render depth and semantic maps

Note this step is not required for training feature Gaussians (Stage 1) or fine-tuning (Stage 2). This step is only needed when we perform linear probing evaluation on ScanNet++. 

We follow [ScanNet++ Toolkit](https://github.com/scannetpp/scannetpp) to generate 2D semantic and depth label by rendering 3D mesh.

Processed scannetpp dataset with annotations to be used for linear probing can be downloaded from [here](https://drive.google.com/file/d/18BGnCzk51nv79M-SiJ6ezX2WOTcQDZyi/view?usp=sharing).
