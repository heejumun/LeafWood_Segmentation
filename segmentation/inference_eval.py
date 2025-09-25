import argparse
import os
import torch
import numpy as np
import importlib
from pathlib import Path
from tqdm import tqdm
from dataset_eval_inference import LeafWoodDataset

# Optional: visualization (remove if not used)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # update as needed


def denormalize(points, centroid, scale_factor):
    """
    Restore normalized point cloud coordinates to the original scale.

    Args:
        points (numpy array): Normalized point cloud (N, 3).
        centroid (numpy array): Centroid used for normalization (3,).
        scale_factor (float): Scale factor used for normalization.

    Returns:
        numpy array: Denormalized point cloud (N, 3).
    """
    return points * scale_factor + centroid


def inplace_relu(m):
    """Convert all ReLU layers to inplace=True."""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """One-hot encode tensor."""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def save_points_to_csv(points, gt_labels, pred_labels, file_dir, fn, class_name):
    """
    Save point cloud with ground truth and predicted labels into a CSV file.

    Args:
        points (numpy array): Point cloud coordinates (N, 3).
        gt_labels (numpy array): Ground truth labels (N,).
        pred_labels (numpy array): Predicted labels (N,).
        file_dir (Path): Output directory.
        fn (str): File name (without extension).
        class_name (torch.Tensor): Class index tensor.
    """
    points = points.squeeze()
    gt_labels = gt_labels.squeeze()
    pred_labels = pred_labels.squeeze()

    if pred_labels.ndim == 1:
        pred_labels = pred_labels.reshape(-1, 1)

    if gt_labels.ndim == 1:
        gt_labels = gt_labels.reshape(-1, 1)

    data = np.hstack((points, gt_labels, pred_labels))

    if len(points) > 0:
        file = file_dir / f'{str(class_name.item()).zfill(8)}/{fn}.csv'
        os.makedirs(file.parent, exist_ok=True)
        np.savetxt(file, data, delimiter=',')
        print(f"Saved CSV to {file}")


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
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_dir = Path("outputs/segmentation_csv")
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

    # Remove "module." prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    classifier.load_state_dict(new_state_dict)
    print('Model loaded from', args.ckpt_path)

    # Load dataset
    test_dataset = LeafWoodDataset(
        root=args.data_root,
        npoints=args.npoint,
        split='inference',
        normal_channel=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f'Total test samples: {len(test_dataset)}')

    # Inference loop
    with torch.no_grad():
        for batch_id, (points, label, target, centroid, m, fn) in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            points = points.float().cuda()
            label = label.long().cuda()

            # Inference
            seg_pred = classifier(points, to_categorical(label, num_classes))  # (B, N, num_parts)
            pred_labels = seg_pred.argmax(dim=-1).cpu().numpy()  # (B, N)

            # Save results
            for i in range(points.size(0)):
                current_points = points[i].cpu().numpy()  # (N, 3)
                current_pred = pred_labels[i]             # (N,)

                # Denormalize points
                denormalized_points = denormalize(
                    current_points,
                    centroid[i].cpu().numpy(),
                    m[i].cpu().numpy()
                )

                # Save results to CSV
                extracted_name = os.path.splitext(os.path.basename(fn[1][0]))[0]
                save_points_to_csv(
                    denormalized_points,
                    target[i].cpu().numpy(),
                    current_pred,
                    file_dir,
                    extracted_name,
                    label
                )


if __name__ == '__main__':
    args = parse_args()
    main(args)
