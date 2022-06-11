import joblib
import os
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch

import numpy as np

super_categories = ['background','Aluminium foil', 'Battery', 'Blister pack', 'Bottle', 
                    'Bottle cap', 'Broken glass', 'Can', 'Carton', 'Cup', 'Food waste', 
                    'Glass jar', 'Lid', 'Other plastic', 'Paper', 'Paper bag', 'Plastic bag & wrapper', 
                    'Plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 
                    'Scrap metal', 'Shoe', 'Squeezable tube', 'Straw', 'Styrofoam piece',
                    'Unlabeled litter', 'Cigarette']
classes_map = {cat: idx for idx, cat in enumerate(super_categories)}

def get_dataloader(set_name):

    dir = os.listdir(f'./data/split_dataset/{set_name}/')
    dir_len = len([d for d in dir if d.startswith(set_name+'_image')])
    if set_name == 'train':
        dir_len = int(dir_len*0.30)

    if set_name == 'val':
        dir_len = int(dir_len*0.50)

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
    
    print(len(labels))
    background_labels = [idx for idx, lab in enumerate(labels) if lab=='background']
    object_labels = [idx for idx, lab in enumerate(labels) if lab!='background']
   
    target_back_or_not = []
    for i in labels:
        if i == 'background': target_back_or_not.append(0)
        else: target_back_or_not.append(1)
    target_back_or_not = np.array(target_back_or_not)
    print('target train 0/1: {}/{}'.format(
        len(np.where(target_back_or_not == 0)[0]), len(np.where(target_back_or_not == 1)[0])))

    #class_sample_count = np.array(
    #    [len(np.where(target_back_or_not == t)[0]) for t in np.unique(target_back_or_not)])

    #weight = 1. / class_sample_count
    #samples_weight = np.array([weight[t] for t in target_back_or_not])

    samples_weight = []
    for t in target_back_or_not:
        if t == 0: samples_weight.append(1)
        else: samples_weight.append(16)

    samples_weight = torch.from_numpy(np.array(samples_weight))
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    labels = np.array([np.array([int(classes_map[label])]) for label in labels])

    data = np.array(data, dtype=np.float32)

    data = torch.Tensor(data) # transform to torch tensor
    labels = torch.Tensor(labels)

    idxs = (labels == classes_map['background']).nonzero(as_tuple=True)[0]
    print(idxs)

    dataset = TensorDataset(data, labels) # create your datset
    dataloader = DataLoader(dataset) # create your dataloader

    return dataloader

val_loader = get_dataloader('val')
#test_loader = get_dataloader('test')
#train_loader = get_dataloader('train')


