import numpy as np
import os
import cv2

from selective_search import selective_search

if __name__ == '__main__':

    for batch in range(15):
        print(f"Searching for bounding boxes in batch {batch}...")
        data_dir = f'./data/batch_{batch+1}/'
        
        if not os.path.exists('./data/proposals/'):
            os.makedirs('./data/proposals/')

        files =  os.listdir(data_dir)
        for file in files:
            image = cv2.imread(data_dir+file)
            image = cv2.resize(image, (224, 224))
            
            boxes = selective_search(image, mode='single', random_sort=False)
            patches = np.array([image[box[0]:box[2], box[1]:box[3], :] for box in boxes], dtype='object')
            
            np.save(f"./data/proposals/{file.split('.')[0]}", patches)