# Leaf-Wood Segmentation
Leaf wood pointcloud separation algorithm.

## Pretrained Models
Pretrained models:
| dataset  | url|
| --- | --- | 
| ShapeNet-55 | [[Google Drive](https://drive.google.com/file/d/1l16Xn5qD5f9gJwOd4XR_WFBRvHyp7jLf/view?usp=drive_link)]|
| ShapeNet-55 + RW-1 | [[Google Drive](https://drive.google.com/file/d/1v_bwFVXuG2mmLp6Y9iiMhqloCllP0SG6/view?usp=drive_link)]| 
| ShapeNet-55 + RW-1 + RW-2 | [[Google Drive](https://drive.google.com/file/d/1EHzpmOqrIVn05sn2xegr2l8W1YKWHp_-/view?usp=drive_link)]|

Finetuned Leaf-Wood Segmentation models:
| dataset  | url| performance |
| --- | --- |  --- |
| ShapeNet-55 | [[Google Drive](https://drive.google.com/file/d/1WzERLlbSwzGOBybzkjBrApwyVMTG00CJ/view?usp=sharing)] | Acc = 0.90, mIoU = 0.74|
| ShapeNet-55 + RW-1 | [[Google Drive](https://drive.google.com/file/d/1vu3vm90WFOefAegkmw1gRklFrymoP_js/view?usp=drive_link)] | Acc = 0.91, mIoU = 0.77| 
| ShapeNet-55 + RW-1 + RW-2 | [[Google Drive](https://drive.google.com/file/d/11AjbHQhVzzIKXKvCKubpU0u6cTGc38-r/view?usp=drive_link)] | Acc = 0.86, mIoU = 0.63| 

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, Earth Mover's Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
    1. cd into extensions/Module (extensions/chamfer_dist)
    2. run `python setup.py install`

# EMD
    1. cd into extensions/Module (extensions/emd)
    2. run `python setup.py install`

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset

Our datasets for pretraining and finetuning are provided in [DATASET.md](./DATASET.md).

## To pretrain Point-M2AE model

### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config <config_file_path> \
    --exp_name <exp_name> 

# resume model
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config <config_file_path> \
    --exp_name <exp_name> \
    --resume
```

## To use Segmentation Algorithm

### Evaluation and Inference

To segment leaf and wood of individual tree point clouds

```
# Evaluation (Point clouds with ground truth labels)
cd segmentation

python inference_eval.py \
    --data_root <path> \
    --ckpt_path <model checkpoint path> \
    --output_dir <dir> \
    --model Point_M2AE_SEG

# Inference (Point Clouds without ground truth labels)
cd segmentation 

python inference_real.py \
    --data_root <path> \
    --ckpt_path <model checkpoint path> \
    --output_dir <dir> \
    --model Point_M2AE_SEG
    [--return_features] -> if you want to get features for each points
```

To preprocess your point clouds to fit the input format

`segmentation/data_preprocessing.ipynb` provides preprocessing code for the following : 
(1) Voxelization (2) File Direction .json creation (3) Large file splitting and (4) Small file deletion.


### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_leafwood.py \
    --model Point_M2AE_SEG \ 
    --log_dir <dir> \
    --ckpts <pretrained_ckpts_path> \
    --distributed 

# resume model 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_leafwood.py \
    --model Point_M2AE_SEG \ 
    --log_dir <dir> \
    --ckpts <last_ckpts_path> \
    --distributed \
    --resume 
```
