# Leaf-Wood Segmentation
Leaf wood pointcloud separation algorithm.

## Pretrained Models
Pretrained models:
| dataset  | url|
| --- | --- | 
| ShapeNet-55 | [[Google Drive](https://drive.google.com/file/d/1ooQcI_aNFRvav9GKhITU6LsEGJ_1aSmT/view?usp=sharing)]|
| ShapeNet-55 + KR | [[Google Drive](https://drive.google.com/file/d/1NWqczbgAWMXLMpC9NtOfC6j5EFJr9Mab/view?usp=sharing)]|

Finetuned Leaf-Wood Segmentation models:
| dataset  | url| performance |
| --- | --- |  --- |
| Randomly-initialized | [[Google Drive](https://drive.google.com/file/d/1mDXFo97AvUbWBTzyr3bbtgD3evUBksGT/view?usp=sharing)] | Acc = 0.92, mIoU = 0.78|
| ShapeNet-pretrained | [[Google Drive](https://drive.google.com/file/d/1LoKmk9nvh-p00SwnmptvY5_uZUWKizF5/view?usp=sharing)] | Acc = 0.92, mIoU = 0.80| 
| ShapeNet + KR-pretrained | [[Google Drive](https://drive.google.com/file/d/16jcfxkEbgtsTPbCXMFLPo86AW2-vWWGl/view?usp=sharing)] | Acc = 0.93, mIoU = 0.82| 

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

### Pretraining

To pretrain a point cloud completion model from scratch, run:

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

### Finetuning

To fine-tune a leaf-wood segmentation model from scratch, run:

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


