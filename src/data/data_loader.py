import random

import numpy as np
import torch
from torchvision import datasets, transforms

from src.common.paths import DATASETS_DIR, SEED


def seed_worker(worker_id):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def get_loader(name_dataset, batch_size, train=True, seed=42, num_workers=4):
    if seed != SEED:
        raise ValueError('DataLoader seed must be 42')

    # Computed with compute_mean_std.py
    mean_std = {
        'amazon': {
            'mean': [0.79235494, 0.7862071 , 0.78418255],
            'std':  [0.31496558, 0.3174693 , 0.3193569 ]
        },
        'dslr': {
            'mean': [0.47086468, 0.44865608, 0.40637794],
            'std':  [0.20395322, 0.19204104, 0.1996422 ]
        },
        'webcam': {
            'mean': [0.6119875 , 0.6187739 , 0.61730677],
            'std':  [0.25063968, 0.25554898, 0.25773206]
        }
    }

    data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[name_dataset]['mean'],
                                 std=mean_std[name_dataset]['std'])
        ])

    dataset = datasets.ImageFolder(root=DATASETS_DIR / name_dataset / 'images',
                                   transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=num_workers,
                                                 drop_last=train,
                                                 generator=torch.Generator().manual_seed(seed),
                                                 worker_init_fn=seed_worker,
                                                 persistent_workers=num_workers > 0,
                                                 pin_memory=torch.cuda.is_available())
    return dataset_loader

