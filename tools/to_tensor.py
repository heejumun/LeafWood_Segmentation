import torch
import numpy as np

def collate_fn(batch, npoints=8192):
    batch = [item for item in batch if item is not None]
    model_ids, points, hyper = zip(*batch)

    padded_points = []

    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)
    hyper_tensor = torch.tensor(hyper)

    return model_ids, point_tensor, hyper_tensor

def collate_fn_cyl(batch, npoints=8192):
    batch = [item for item in batch if item is not None]
    model_ids, points, hyper = zip(*batch)

    padded_points = []

    for point in points:
        if len(point) < npoints:
            pad_size = npoints - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = point[:npoints]
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)
    hyper_tensor = torch.tensor(hyper)

    return model_ids, point_tensor, hyper_tensor

def random_sample(pc, num, per_num):
        permutation = np.arange(per_num)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

def collate_fn_up(batch, npoints=32768):
    batch = [item for item in batch if item is not None]
    model_ids, points, gt_points = zip(*batch)

    padded_points = []
    gt_padded_points = []
    for point in points:
        if len(point) < 8192:
            pad_size = 8192 - len(point)
            point_data = np.pad(point, ((0, pad_size), (0, 0)), mode='constant')
        else:
            point_data = random_sample(point,8192,len(point))
        padded_points.append(torch.tensor(point_data, dtype=torch.float32))

    point_tensor = torch.stack(padded_points)
    

    for gt_point in gt_points:
        if len(gt_point) < 32768:
            gtpad_size = 32768 - len(gt_point)
            gtpoint_data = np.pad(gt_point, ((0, gtpad_size), (0, 0)), mode='constant')
        else:
            
            gtpoint_data = random_sample(gt_point, 32768, len(gt_point))
        gt_padded_points.append(torch.tensor(gtpoint_data, dtype=torch.float32))

    gt_point_tensor = torch.stack(gt_padded_points)

    return model_ids, point_tensor, gt_point_tensor


    
        