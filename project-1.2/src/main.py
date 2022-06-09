import numpy as np
import os
import glob
import cv2

import skimage.io
from selective_search import selective_search

if __name__ == '__main__':

    for batch in range(1, 15):
        data_dir = glob.glob(f'/dtu/datasets1/02514/data_wastedetection/batch_{batch}/')
        
        if not os.path.exists('./data/proposals/'):
            os.makedirs('./data/proposals/')

        files =  os.listdir(data_dir)
        for file in files:
            image = cv2.imread(data_dir+file)
            image = cv2.resize(image, (224, 224))
            
            boxes = selective_search(image, mode='single', random_sort=False)
            patches = np.array([image[box[0]:box[2], box[1]:box[3], :] for box in boxes], dtype='object')
            
            np.save(f"./data/proposals/{file.split('.')[0]}", patches)