import argparse
import os
import torch
import numpy as np
import importlib
from pathlib import Path
from tqdm import tqdm
from dataset_real_inference import LeafWoodDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def denormalize(points, centroid, scale_factor):
    """Normalize된 point cloud 데이터를 원래 좌표로 복원"""
    return points * scale_factor + centroid

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.view(-1).cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def save_points_to_csv(points, pred_labels, file_dir, fn, class_name):
    """Point cloud와 라벨을 CSV 저장"""
    points = points.squeeze()
    pred_labels = pred_labels.squeeze()

    if pred_labels.ndim == 1:
        pred_labels = pred_labels.reshape(-1, 1)

    data = np.hstack((points, pred_labels))

    if len(points) > 0:
        file = file_dir / f'{str(class_name.item()).zfill(8)}/{fn}.csv'
        os.makedirs(file.parent, exist_ok=True)
        np.savetxt(file, data, delimiter=',')
        print(f"Saved points to {file}")


def parse_args():
    parser = argparse.ArgumentParser('Inference')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save the results')
    parser.add_argument('--npoint', type=int, default=32768, help='Number of points to sample')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--model', type=str, default='Point_M2AE_SEG', help='Model name')
    parser.add_argument('--return_features', action='store_true', help='If set, save intermediate features')
    return parser.parse_args()


def main(args):
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convs4_dir = output_dir / "convs4"
    softmax_dir = output_dir / "softmax"
    convs4_dir.mkdir(parents=True, exist_ok=True)
    softmax_dir.mkdir(parents=True, exist_ok=True)

    # Create segmentation results dir
    file_dir = Path('/bess25/heeju/DATA/Final/Model_B_inf')
    file_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    num_part = 4
    num_classes = 2
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.Point_M2AE_SEG(num_part).cuda()
    classifier.apply(inplace_relu)
    classifier.eval()

    # Hook dictionary
    features = {}
    if args.return_features:
        def hook_fn(module, input, output):
            features["convs4"] = output.detach().cpu().numpy()
        classifier.convs4.register_forward_hook(hook_fn)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

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
        for batch_id, (points, label, centroid, m, fn) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            points = points.float().cuda()
            label = label.long().cuda()

            # Inference (log_softmax output)
            seg_pred = classifier(points, to_categorical(label, num_classes))  # (B, N, num_parts)
            pred_labels = seg_pred.argmax(dim=-1).cpu().numpy()

            for i in range(points.size(0)):
                current_points = points[i].cpu().numpy()
                current_labels = pred_labels[i]

                denormalized_points = denormalize(
                    current_points,
                    centroid[i].cpu().numpy(),
                    m[i].cpu().numpy()
                )

                extracted_name = os.path.splitext(os.path.basename(fn[1][0]))[0]

                # Save segmentation results
                save_points_to_csv(
                    denormalized_points,
                    current_labels,
                    file_dir,
                    extracted_name,
                    label
                )

                # Save features if requested
                if args.return_features:
                    # (1) convs4 raw logits 저장
                    if "convs4" in features:
                        logits = features["convs4"].squeeze()
                        if logits.ndim == 3:  # (B, C, N)
                            logits = logits[0].T  # (N, C)
                        np.savetxt(convs4_dir / f"{extracted_name}.csv", logits, delimiter=',')
                        print(f"Saved convs4 logits to {convs4_dir / (extracted_name + '.csv')}")

                    # (2) log_softmax 출력 저장
                    logsoftmax_out = seg_pred[i].cpu().numpy()  # (N, num_parts)
                    np.savetxt(softmax_dir / f"{extracted_name}.csv", logsoftmax_out, delimiter=',')
                    print(f"Saved log_softmax to {softmax_dir / (extracted_name + '.csv')}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
