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


def random_sample(pc, num, per_num):
        permutation = np.arange(per_num)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc
    
        