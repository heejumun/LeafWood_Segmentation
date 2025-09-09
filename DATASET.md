## Dataset

### Overall Dataset:

```
│M2AE_LWSEG/
├──cfgs/
├──data/
│   ├──M2AE-Pretrain/
│   │   ├──ShapeNet55-34/
│   │   ├──ShapeNet_RW1/ 
│   │   ├──ShapeNet_RW12/
│   │   ├──ModelNet40_svm/ 
│   │   ├──ModelNet_RW1_svm/
│   │   ├──ModelNet_RW12_svm/ 
│   ├──M2AE-Finetune/
│   │   ├──Voxel_LW/ 
├──datasets/
├──.......
```

### ShapeNet55/34 Dataset:

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.csv
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.csv
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

### ShapeNet_RW Dataset:

```
│ShapeNet_RW_{i}/
├──shapenet_rw{i}/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.csv
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.csv
│  ├── .......
│  ├── 99999999-16780000.csv (additional tree category)
├──ShapeNet_rw{i}_txt/
│  ├── train.txt
│  └── test.txt
```

### ModelNet40_svm Dataset:

```
│modelnet40_ply_h5_2048/
├──ply_data_train0.h5
├──ply_data_train1.h5
├── .......
├──ply_data_test0.h5
└──ply_data_test1.h5

```

### ModelNet_RW{i} Dataset:

```
│modelnet_tree_h5_2048/
├──ply_data_train_tree_0.h5
├──ply_data_train_tree_1.h5
├── .......
├──ply_data_test_tree_0.h5
└──ply_data_test_tree_1.h5

```

Download: Please download the data from [here](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md).

### Voxel_LW Dataset:

```
│Voxel_LW/
├──22222222/ (needleleaf)
│  ├── 05660001.csv OOOOXXXX (O: tree instance, X: voxel id)
│  ├── 05660002.csv
│  ├── .......
├──33333333/ (broadleaf)
│  ├── 00180001.csv
│  ├── 00180002.csv
│  ├── .......
├──classified_json_grouped
│  ├── leafwood_data_inference.json
│  ├── leafwood_data_train.json
│  ├── leafwood_data_val.json
│  └── leafwood_data_test.json
└──synsetoffset2category.txt

```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 