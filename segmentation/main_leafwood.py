"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
sys.path.append('/esail4/heeju/Point-M2AE')
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

sys.path.append("./")
seg_classes = {'NL': [0, 1], 'BL' : [2, 3]}
seg_label_to_cat = {}  # {0:Tree, 1:Tree}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """1-hot 인코딩"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def setup_distributed():
    """분산 학습 환경 초기화 (backend: nccl)"""
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank


def freeze_except_coarse_and_dense(model):
    for name, param in model.named_parameters():
        # "propagations"와 "convs"에 해당하는 레이어만 학습
        if "propagations" not in name and "convs" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    parser.add_argument('--npoint', type=int, default=32768, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='/esail4/heeju/REGRESSION/Point-M2AE/data/segmentation', help='data root')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--distributed', action='store_true', default=False, help='Use DistributedDataParallel')
    return parser.parse_args()


def main(args):
    if args.distributed:
        local_rank = setup_distributed()
    else:
        local_rank = 0

    # 오직 rank 0에서만 로그 출력하도록 helper 함수 정의
    def log_string(s):
        if local_rank == 0:
            logger.info(s)
            print(s)

    '''CREATE DIR'''
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()  # (필요 시)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if local_rank == 0:
        log_string('PARAMETER ...')
        log_string(args)

    root = args.root

    TRAIN_DATASET = LeafWoodDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    TEST_DATASET = LeafWoodDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

    # 분산 환경일 경우, 전체 GPU 수에 따라 배치 사이즈 조정
    world_size = dist.get_world_size() if args.distributed else 1
    args.batch_size = args.batch_size // world_size

    # DDP를 위한 데이터 샘플러 적용
    train_sampler = DistributedSampler(TRAIN_DATASET) if args.distributed else None
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=10,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=lambda x: collate_fn(x, npoints=args.npoint)
    )

    test_sampler = DistributedSampler(TEST_DATASET) if args.distributed else None
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        sampler=test_sampler,
        collate_fn=lambda x: collate_fn(x, npoints=args.npoint)
    )

    if local_rank == 0:
        log_string("The number of training data is: %d" % len(TRAIN_DATASET))
        log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 2
    num_part = 4

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    if local_rank == 0:
        shutil.copy(f'./{args.model}.py', str(exp_dir))
    classifier = MODEL.Point_M2AE_SEG(num_part).cuda()

    if args.distributed:
        classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)

    criterion = MODEL.get_loss().cuda()

    classifier.apply(inplace_relu)
    # freeze_except_coarse_and_dense(classifier)
    if local_rank == 0:
        print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

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

    if args.resume and args.ckpts is not None:  # Resume training
        if local_rank == 0:
            log_string(f"Resuming training from checkpoint: {args.ckpts}")
        checkpoint = torch.load(args.ckpts, map_location=f'cuda:{local_rank}')
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if local_rank == 0:
            log_string(f"Resumed training from epoch {start_epoch}")
    elif args.ckpts is not None:  # Load weights only
        if local_rank == 0:
            log_string(f"Loading model weights from checkpoint: {args.ckpts}")
        checkpoint = torch.load(args.ckpts, map_location=f'cuda:{local_rank}')
        classifier.load_state_dict(checkpoint['base_model'], strict=False)
        start_epoch = 0
        if local_rank == 0:
            log_string("Loaded model weights. Starting training from epoch 0.")
    else:
        start_epoch = 0
        if local_rank == 0:
            log_string("No checkpoint provided. Starting training from scratch.")

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)  # 각 epoch마다 데이터 순서를 섞음
        mean_correct = []

        if local_rank == 0:
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier.train()
        loss_batch = []
        num_iter = 0
        '''한 epoch 학습'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9, disable=(local_rank != 0)):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

            seg_pred = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            mask = (target != -99999)  # 유효한 포인트만 선택

            seg_pred_masked = seg_pred[mask]
            target_masked = target[mask]


            loss = criterion(seg_pred_masked, target_masked)

            # 정확도 계산
            pred_choice = seg_pred_masked.data.max(1)[1]
            correct = pred_choice.eq(target_masked.data).cpu().sum()
            mean_correct.append(correct.item() / mask.sum().item())

            if torch.isnan(seg_pred_masked).any() or torch.isinf(seg_pred_masked).any():
                print("seg_pred_masked contains NaN or Inf values!")
            if torch.isnan(target_masked).any() or torch.isinf(target_masked).any():
                print("target_masked contains NaN or Inf values!")

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
        # 분산 시 GPU별 train accuracy 합산 후 평균 내기
        if args.distributed:
            train_acc_tensor = torch.tensor(train_instance_acc).cuda()
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            train_instance_acc = train_acc_tensor.item() / world_size

        loss1 = np.mean(loss_batch)
        if local_rank == 0:
            log_string('Train accuracy is: %.5f' % train_instance_acc)
            log_string('Train loss: %.5f' % loss1)
            log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        # ======================= Validation =======================
        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            total_seen_class = np.zeros(num_part, dtype=np.int64)
            total_correct_class = np.zeros(num_part, dtype=np.int64)
            # 각 카테고리별 IoU 합과 카운트를 저장 (추후 평균 계산)
            iou_sum_per_cat = {cat: 0.0 for cat in seg_classes.keys()}
            iou_count_per_cat = {cat: 0 for cat in seg_classes.keys()}

            classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9, disable=(local_rank != 0)):
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

                # correct = np.sum(cur_pred_val == target_np)
                # total_correct += correct
                # total_seen += (cur_batch_size * NUM_POINT)

                valid_mask = target_np != -99999
                total_correct += np.sum((cur_pred_val == target_np) & valid_mask)
                total_seen += np.sum(valid_mask)

                for l in range(num_part):
                    mask = (target_np == l) & valid_mask
                    total_seen_class[l] += np.sum(target_np == l)
                    total_correct_class[l] += np.sum((cur_pred_val == l) & mask)


                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target_np[i, :]

                    # 유효한 포인트 마스크 (padding 제외)
                    valid_mask = segl != -99999
                    segp = segp[valid_mask]
                    segl = segl[valid_mask]

                    if len(segl) == 0:  # all padding
                        continue

                    cat = seg_label_to_cat[segl[0]]
                    part_ious = []
                    for l in seg_classes[cat]:
                        union = np.sum((segl == l) | (segp == l))
                        if union == 0:
                            iou = 1.0
                        else:
                            iou = np.sum((segl == l) & (segp == l)) / float(union)
                        part_ious.append(iou)

                    iou_sample = np.mean(part_ious)
                    iou_sum_per_cat[cat] += iou_sample
                    iou_count_per_cat[cat] += 1


            # 분산 학습 시 각 GPU에서 계산한 지표 합산
            if args.distributed:
                device = torch.device("cuda", local_rank)
                total_correct_tensor = torch.tensor(total_correct).to(device)
                total_seen_tensor = torch.tensor(total_seen).to(device)
                dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_seen_tensor, op=dist.ReduceOp.SUM)
                total_correct = total_correct_tensor.item()
                total_seen = total_seen_tensor.item()

                total_correct_class_tensor = torch.tensor(total_correct_class).to(device)
                total_seen_class_tensor = torch.tensor(total_seen_class).to(device)
                dist.all_reduce(total_correct_class_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_seen_class_tensor, op=dist.ReduceOp.SUM)
                total_correct_class = total_correct_class_tensor.cpu().numpy()
                total_seen_class = total_seen_class_tensor.cpu().numpy()

                for cat in seg_classes.keys():
                    sum_tensor = torch.tensor(iou_sum_per_cat[cat]).to(device)
                    count_tensor = torch.tensor(iou_count_per_cat[cat]).to(device)
                    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                    iou_sum_per_cat[cat] = sum_tensor.item()
                    iou_count_per_cat[cat] = count_tensor.item()

            shape_ious = {}
            for cat in seg_classes.keys():
                if iou_count_per_cat[cat] > 0:
                    shape_ious[cat] = iou_sum_per_cat[cat] / iou_count_per_cat[cat]
                else:
                    shape_ious[cat] = 0.0
            mean_shape_ious = np.mean(list(shape_ious.values()))
            global_iou_sum = sum(iou_sum_per_cat.values())
            global_count = sum(iou_count_per_cat.values())
            instance_avg_iou = global_iou_sum / global_count if global_count > 0 else 0.0

            test_metrics = {}
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = instance_avg_iou

        if local_rank == 0:
            log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
            if test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_avg_iou': test_metrics['class_avg_iou'],
                    'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
            if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
                best_inctance_avg_iou = test_metrics['inctance_avg_iou']
            log_string('Best accuracy is: %.5f' % best_acc)
            log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
            log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
