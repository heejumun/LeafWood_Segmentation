"""
Author: Benny (modified for public release)
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
sys.path.append('./')  # adjust module path if needed
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import LeafWoodDataset
from collate_fn import collate_fn

# segmentation classes
seg_classes = {'NL': [0, 1], 'BL': [2, 3]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    """Set inplace=True for all ReLU layers."""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """One-hot encode class labels."""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def setup_distributed():
    """Initialize distributed training with NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Apply weight decay except for bias and norm parameters."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Segmentation Training')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size during training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epochs')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epochs')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log directory')
    parser.add_argument('--npoint', type=int, default=16384, help='number of points')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals as input features')
    parser.add_argument('--ckpts', type=str, default=None, help='checkpoint path')
    parser.add_argument('--root', type=str, default='dataset/LWSEG_Voxel', help='dataset root path')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    parser.add_argument('--distributed', action='store_true', default=False, help='use DistributedDataParallel')
    return parser.parse_args()


def main(args):
    if args.distributed:
        local_rank = setup_distributed()
    else:
        local_rank = 0

    def log_string(s):
        if local_rank == 0:
            logger.info(s)
            print(s)

    # === Create directories ===
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_dir = Path('./log/part_seg')
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir if args.log_dir else timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # === Logging ===
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if local_rank == 0:
        log_string('PARAMETERS ...')
        log_string(args)

    # === Dataset ===
    root = args.root
    TRAIN_DATASET = LeafWoodDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    TEST_DATASET = LeafWoodDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

    world_size = dist.get_world_size() if args.distributed else 1
    args.batch_size = args.batch_size // world_size

    train_sampler = DistributedSampler(TRAIN_DATASET) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=10,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=lambda x: collate_fn(x, npoints=args.npoint)
    )
    test_sampler = DistributedSampler(TEST_DATASET) if args.distributed else None
    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        sampler=test_sampler,
        collate_fn=lambda x: collate_fn(x, npoints=args.npoint)
    )

    if local_rank == 0:
        log_string(f"Training samples: {len(TRAIN_DATASET)}")
        log_string(f"Test samples: {len(TEST_DATASET)}")

    num_classes, num_part = 2, 4

    # === Model ===
    MODEL = importlib.import_module(args.model)
    if local_rank == 0:
        shutil.copy(f'./{args.model}.py', str(exp_dir))
    classifier = MODEL.Point_M2AE_SEG(num_part).cuda()

    if args.distributed:
        classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)

    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    if local_rank == 0:
        print('# parameters:', sum(param.numel() for param in classifier.parameters()))

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05, capturable=True)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epoch,
        t_mul=1,
        lr_min=1e-6,
        decay_rate=0.1,
        warmup_lr_init=1e-6,
        warmup_t=args.warmup_epoch,
        cycle_limit=1,
        t_in_epochs=True
    )

    # === Best metrics ===
    best_acc, best_class_avg_iou, best_instance_avg_iou = 0, 0, 0
    start_epoch, global_epoch = 0, 0

    # === Resume or load checkpoint ===
    if args.resume and args.ckpts is not None:
        checkpoint = torch.load(args.ckpts, map_location=f'cuda:{local_rank}')
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint.get('best_acc', 0)
        best_class_avg_iou = checkpoint.get('best_class_avg_iou', 0)
        best_instance_avg_iou = checkpoint.get('best_instance_avg_iou', 0)
        start_epoch = checkpoint['epoch'] + 1
        log_string(f"Resumed training from epoch {start_epoch}")
    elif args.ckpts is not None:
        checkpoint = torch.load(args.ckpts, map_location=f'cuda:{local_rank}')
        classifier.load_state_dict(checkpoint['base_model'], strict=False)
        log_string("Loaded model weights only. Training from scratch.")
    else:
        log_string("No checkpoint provided. Training from scratch.")

    # === Training loop ===
    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_correct, loss_batch = [], []
        if local_rank == 0:
            log_string(f"Epoch {epoch+1}/{args.epoch}")

        classifier.train()
        num_iter = 0
        for _, (points, label, target) in tqdm(enumerate(train_loader), total=len(train_loader), disable=(local_rank != 0)):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

            seg_pred = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            mask = (target != -99999)

            seg_pred_masked, target_masked = seg_pred[mask], target[mask]
            loss = criterion(seg_pred_masked, target_masked)

            pred_choice = seg_pred_masked.data.max(1)[1]
            correct = pred_choice.eq(target_masked.data).cpu().sum()
            mean_correct.append(correct.item() / mask.sum().item())

            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        scheduler.step(epoch)
        train_instance_acc = np.mean(mean_correct)

        if args.distributed:
            train_acc_tensor = torch.tensor(train_instance_acc).cuda()
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            train_instance_acc = train_acc_tensor.item() / world_size

        if local_rank == 0:
            log_string(f"Train acc: {train_instance_acc:.5f}, loss: {np.mean(loss_batch):.5f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        # === Validation ===
        with torch.no_grad():
            total_correct, total_seen = 0, 0
            total_seen_class = np.zeros(num_part, dtype=np.int64)
            total_correct_class = np.zeros(num_part, dtype=np.int64)
            iou_sum_per_cat = {cat: 0.0 for cat in seg_classes.keys()}
            iou_count_per_cat = {cat: 0 for cat in seg_classes.keys()}
            classifier.eval()

            for _, (points, label, target) in tqdm(enumerate(test_loader), total=len(test_loader), disable=(local_rank != 0)):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

                seg_pred = classifier(points, to_categorical(label, num_classes))
                cur_pred_val_logits = seg_pred.cpu().data.numpy()
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT), dtype=np.int32)
                target_np = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target_np[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], axis=1) + seg_classes[cat][0]

                valid_mask = target_np != -99999
                total_correct += np.sum((cur_pred_val == target_np) & valid_mask)
                total_seen += np.sum(valid_mask)

                for l in range(num_part):
                    mask = (target_np == l) & valid_mask
                    total_seen_class[l] += np.sum(target_np == l)
                    total_correct_class[l] += np.sum((cur_pred_val == l) & mask)

                for i in range(cur_batch_size):
                    segp, segl = cur_pred_val[i, :], target_np[i, :]
                    valid_mask = segl != -99999
                    segp, segl = segp[valid_mask], segl[valid_mask]
                    if len(segl) == 0:
                        continue
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = []
                    for l in seg_classes[cat]:
                        union = np.sum((segl == l) | (segp == l))
                        iou = 1.0 if union == 0 else np.sum((segl == l) & (segp == l)) / float(union)
                        part_ious.append(iou)
                    iou_sum_per_cat[cat] += np.mean(part_ious)
                    iou_count_per_cat[cat] += 1

            if args.distributed:
                device = torch.device("cuda", local_rank)
                total_correct_tensor = torch.tensor(total_correct).to(device)
                total_seen_tensor = torch.tensor(total_seen).to(device)
                dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_seen_tensor, op=dist.ReduceOp.SUM)
                total_correct = total_correct_tensor.item()
                total_seen = total_seen_tensor.item()

            shape_ious = {cat: (iou_sum_per_cat[cat] / iou_count_per_cat[cat] if iou_count_per_cat[cat] > 0 else 0.0)
                          for cat in seg_classes.keys()}
            mean_shape_ious = np.mean(list(shape_ious.values()))
            instance_avg_iou = sum(iou_sum_per_cat.values()) / max(sum(iou_count_per_cat.values()), 1)

            test_metrics = {
                'accuracy': total_correct / float(total_seen),
                'class_avg_accuracy': np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float)),
                'class_avg_iou': mean_shape_ious,
                'instance_avg_iou': instance_avg_iou
            }

        if local_rank == 0:
            log_string(f"Epoch {epoch+1} Test Accuracy: {test_metrics['accuracy']:.5f}, "
                       f"Class avg IoU: {test_metrics['class_avg_iou']:.5f}, "
                       f"Instance avg IoU: {test_metrics['instance_avg_iou']:.5f}")

            # save best
            if test_metrics['instance_avg_iou'] >= best_instance_avg_iou:
                savepath = str(checkpoints_dir / 'best_model.pth')
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_avg_iou': test_metrics['class_avg_iou'],
                    'instance_avg_iou': test_metrics['instance_avg_iou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'best_class_avg_iou': best_class_avg_iou,
                    'best_instance_avg_iou': best_instance_avg_iou
                }
                torch.save(state, savepath)
                log_string(f"Best model saved at {savepath}")

            # save last
            savepath = str(checkpoints_dir / 'ckpt_last.pth')
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_class_avg_iou': best_class_avg_iou,
                'best_instance_avg_iou': best_instance_avg_iou
            }
            torch.save(state, savepath)

            # update bests
            best_acc = max(best_acc, test_metrics['accuracy'])
            best_class_avg_iou = max(best_class_avg_iou, test_metrics['class_avg_iou'])
            best_instance_avg_iou = max(best_instance_avg_iou, test_metrics['instance_avg_iou'])
            log_string(f"Best acc: {best_acc:.5f}, Best class IoU: {best_class_avg_iou:.5f}, "
                       f"Best instance IoU: {best_instance_avg_iou:.5f}")

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
