import joblib
import os
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
import torch

import numpy as np

def get_dataloader(set_name):

    dir = os.listdir(f'./data/split_dataset/{set_name}/')
    dir_len = len([d for d in dir if d.startswith(set_name+'_image')])

    data_sep = []
    data, labels = [], []
    for idx in tqdm(range(dir_len)):
        img_prefix = f'{set_name}_image_{idx}.pkl'
        lab_prefix = f'{set_name}_labels_{idx}.pkl'
        
        img_proposals = joblib.load(f'./data/split_dataset/{set_name}/'+img_prefix)
        data_sep.append(img_proposals)
        data.extend(img_proposals)

        lab_proposals = joblib.load(f'./data/split_dataset/{set_name}/'+lab_prefix)
        labels.extend(lab_proposals)
            
    classes_map = dict(zip(set(labels), range(len(labels))))
    labels = np.array([np.array([int(classes_map[label])]) for label in labels])

    data = np.array(data, dtype=np.float32)
    data = torch.Tensor(data) # transform to torch tensor
    labels = torch.Tensor(labels)

    dataset = TensorDataset(data, labels) # create your datset
    dataloader = DataLoader(dataset) # create your dataloader

    return dataloader

val_loader = get_dataloader('val')
test_loader = get_dataloader('test')
exit()
train_loader = get_dataloader('train')
