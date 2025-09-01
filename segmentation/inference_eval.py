import argparse
import os
import torch
import numpy as np
import importlib
from pathlib import Path
from tqdm import tqdm
from dataset_eval_inference import LeafWoodDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def denormalize(points, centroid, scale_factor):
    """
    Normalize된 point cloud 데이터를 원래 좌표로 복원합니다.

    Args:
        points (numpy array): Normalize된 point cloud 좌표 (N, 3).
        centroid (numpy array): Normalize에 사용된 중심점 (3,).
        scale_factor (float): Normalize에 사용된 스케일 팩터.

    Returns:
        numpy array: 복원된 point cloud 좌표 (N, 3).
    """
    return points * scale_factor + centroid

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def visualize_pointcloud(points, labels, save_path, title="Point Cloud Visualization"):
    """
    Point cloud 데이터를 시각화하고 이미지를 저장합니다.

    Args:
        points (numpy array): Point cloud 좌표 (N, 3).
        labels (numpy array): Point cloud의 라벨 (N,).
        save_path (str): 결과 이미지를 저장할 경로.
        title (str): 시각화 제목.
    """
    # Color-blind friendly colors
    leaf_color = '#56B4E9'  # Light blue
    wood_color = '#E69F00'  # Orange

    fig = plt.figure(figsize=(20, 8))

    # 잎과 나무를 구분
    leaf_points = points[labels == 1]
    tree_points = points[labels == 0]

    print(leaf_points.shape)
    print(tree_points.shape)

def save_points_to_csv(points, gt_labels, pred_labels, file_dir, fn, class_name):
    """
    Point cloud 데이터를 CSV로 저장합니다.

    Args:
        points (numpy array): Point cloud 좌표 (N, 3).
        labels (numpy array): Point cloud의 라벨 (N,).
        leaf_dir (Path): 잎 포인트 저장 디렉토리.
        wood_dir (Path): 나무 포인트 저장 디렉토리.
        start_idx (int): 저장 파일의 시작 인덱스.
    """
    points = points.squeeze()
    gt_labels = gt_labels.squeeze()
    pred_labels = pred_labels.squeeze()

    if pred_labels.ndim == 1:
        pred_labels = pred_labels.reshape(-1, 1)

    if gt_labels.ndim == 1:
        gt_labels = gt_labels.reshape(-1, 1)
    # 잎과 나무로 분리
    data = np.hstack((points, gt_labels, pred_labels))
    
    # 잎 저장
    if len(points) > 0:
        file = file_dir / f'{str(class_name.item()).zfill(8)}/{fn}.csv'
        np.savetxt(file, data, delimiter=',')
        print(f"Saved Leaf points to {file}")


def parse_args():
    parser = argparse.ArgumentParser('Inference')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save the results')
    parser.add_argument('--npoint', type=int, default=32768, help='Number of points to sample')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--model', type=str, default='Point_M2AE_SEG', help='Model name')
    return parser.parse_args()


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create leaf and wood directories
    file_dir = Path('/esail4/heeju/REGRESSION/Point-M2AE/segmentation2/comb1_full_csv_0826')
    file_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    num_part = 4
    num_classes = 2
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.Point_M2AE_SEG(num_part).cuda()
    classifier.apply(inplace_relu)
    classifier.eval()

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # "module." 프리픽스 제거
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    classifier.load_state_dict(new_state_dict)
    print('Model loaded from', args.ckpt_path)

    # Load dataset
    test_dataset = LeafWoodDataset(root=args.data_root, npoints=args.npoint, split='inference', normal_channel=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f'Total test samples: {len(test_dataset)}')

    with torch.no_grad():
        for batch_id, (points, label, target, centroid, m, fn) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            points = points.float().cuda()
            label = label.long().cuda()
            # Inference
            seg_pred = classifier(points, to_categorical(label, num_classes))  # Shape: (B, N, num_parts)
            pred_labels = seg_pred.argmax(dim=-1).cpu().numpy()  # Shape: (B, N)
            # Save results
            for i in range(points.size(0)):
                current_points = points[i].cpu().numpy() # Shape: (N, 3)
                current_labels = pred_labels[i]  # Shape: (N,)
          
                # Denormalize current points
                denormalized_points = denormalize(
                    current_points,
                    centroid[i].cpu().numpy(),  # Batch에서 i번째 centroid
                    m[i].cpu().numpy()          # Batch에서 i번째 scale factor
                )

                # Visualize and save results
                extracted_name = os.path.splitext(os.path.basename(fn[1][0]))[0]

                # Save leaf and wood points to CSV
                save_points_to_csv(
                    denormalized_points,
                    current_labels,
                    target,
                    file_dir,
                    extracted_name,
                    label
                )

if __name__ == '__main__':
    args = parse_args()
    main(args)
