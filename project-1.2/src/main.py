import numpy as np
import os
from multiprocessing import Pool
from functools import partial
import cv2
from tqdm import tqdm

from selective_search import selective_search


def process_image(file, data_dir, batch_idx):
    print(f"Processing {file}...")
    image = cv2.imread(data_dir+file)
    image = cv2.resize(image, (224, 224))
    
    boxes = np.array(selective_search(image, mode='single', random_sort=False), dtype='object')
    data_dir.split('/')[2]
    np.save(f"./data/proposals/{batch_idx}/{file.split('.')[0]}", boxes)


    """
    patches = []
    for box in boxes:
        patch = image[box[0]:box[2], box[1]:box[3], :]
        patch_res = cv2.resize(patch, (32, 32))
        patches.append(patch_res)
    
    patches = np.array(patches, dtype='object')
    np.save(f"./data/proposals/{file.split('.')[0]}", patches)
    """


if __name__ == '__main__':

    for batch in range(15):
        print(f"Searching for bounding boxes in batch {batch+1}...")
        data_dir = f'./data/batch_{batch+1}/'
        
        if not os.path.exists('./data/proposals/'):
            os.makedirs('./data/proposals/')

        if not os.path.exists(f'./data/proposals/{batch+1}/'):
            os.makedirs(f'./data/proposals/{batch+1}/')

        files = os.listdir(data_dir)
        
        with Pool(processes=4) as pool:
            func = partial(process_image, data_dir=data_dir, batch_idx=batch+1)
            pool.map(func, files)
                