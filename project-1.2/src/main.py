import numpy as np
import os
import json
from copy import deepcopy

from multiprocessing import Pool
from functools import partial

import cv2

from selective_search import selective_search

from sklearn.model_selection import StratifiedShuffleSplit

image_size = 224
cropped_image_size = 64

def get_image_ground_truth(dataset, filename):
    #----------------------------------------------
    # Function to find the ground truth annotation for each resized image 
    #----------------------------------------------

    # Find the image id in the annotation file
    for img in dataset['images']:
        if img['file_name'] == filename:
            my_img = deepcopy(img)

    if my_img['width'] > 512 or my_img['height'] > 512:
        if my_img['width'] > my_img['height']:
            new_width = 512
            new_height = int(my_img['height'] * (512./my_img['width']))
        elif my_img['height'] > my_img['width']:
            new_height = 512
            new_width = int(my_img['width'] * (512./my_img['height']))
        else:
            new_width = 512
            new_height = 512

    w_factor = float(new_width)/my_img['width']
    h_factor = float(new_height)/my_img['height']

    # Finding all the annotations for one image - maybe try to optimize it by using pandas?
    img_annots = {}
    img_annots['id'] = []
    img_annots['bbox'] = []
    img_annots['category_id'] = []
    img_annots['supercategory'] = []
    img_annots['orig_size'] = [my_img['width'], my_img['height']]
    img_annots['new_size'] = [new_width, new_height]

    for annot in dataset['annotations']:
        if annot['image_id'] == my_img['id']:
            #print(annot['id'], annot['bbox'], annot['category_id'])#, dataset['categories']['id'][annot['id']])
            img_annots['id'].append(annot['id'])
            img_annots['bbox'].append([round(annot['bbox'][0]*w_factor,1), round(annot['bbox'][1]*h_factor,1), round(annot['bbox'][2]*w_factor,1), round(annot['bbox'][3]*h_factor,1)])
            img_annots['category_id'].append(annot['category_id'])
            for cat in dataset['categories']:
                if cat['id'] == annot['category_id']:
                    #print(cat['supercategory'])
                    img_annots['supercategory'].append(cat['supercategory'])
                    
    return img_annots
                         
# All the images have to be resized to the standard size and hence the annotation boxes coordinates
# [x, y, w, h] = ann['bbox']

def get_image_proposals(filepath):
    # Get the proposals for an image
    image = cv2.imread(filepath)
    image = cv2.resize(image,(image_size,image_size))
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    prop = ss.process()[:2000]
    return prop


def get_iou(bb1, bb2):
    # bb1 = [x1 y1 w1 h1]
    # bb2 = [x2 y2 w2 h2]
    
    #x1_bb1 = bb1[0]
    #y1_bb1 = bb1[1]
    #x2_bb1 = bb1[0] + bb1[2]
    #y2_bb1 = bb1[1] + bb1[3]
    
    #x1_bb2 = bb2[0]
    #y1_bb2 = bb2[1]
    #x2_bb2 = bb2[0] + bb2[2]
    #y2_bb2 = bb2[1] + bb2[3]
    
    # assuring for proper dimension.
    assert bb1[0] < bb1[0] + bb1[2]
    assert bb1[1] < bb1[1] + bb1[3]
    assert bb2[0] < bb2[0] + bb2[2]
    assert bb2[1] < bb2[1] + bb2[3]
    # calculating dimension of common area between these two boxes.
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # calculating intersection area.
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # individual areas of both these bounding boxes.
    bb1_area = (bb1[2]) * (bb1[3])
    bb2_area = (bb2[2]) * (bb2[3])
    # union area = area of bb1_+ area of bb2 - intersection of bb1 and bb2.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def assign_category_to_proposal(prop, img_annots):
    assignemt_prob = np.zeros([len(prop),len(img_annots['id'])])

    for i in range(len(prop)):
        for j in range(len(img_annots['bbox'])):
            assignemt_prob[i,j] = get_iou(np.array(prop[i]),np.array(img_annots['bbox'][j]))
            
    prop_categories = []

    for i in range(len(prop)):
        if not (assignemt_prob[i,:] > 0.5).any(): 
            prop_categories.append('background')
        else:
            index = np.argmax(assignemt_prob[i,:])
            prop_categories.append(img_annots['supercategory'][index])
            
    return prop_categories

def crop_images_to_proposals(filepath, prop, new_image_size):
    image = cv2.imread(filepath)
    image = cv2.resize(image,(image_size,image_size))
    cropped_resized_images = []
    for box in prop:
        cropped_image = image[int(box[1]):int(box[1]+box[3]),int(box[0]):int(box[0] + box[2])]
        try:
            cropped_resized_images.append(cv2.resize(cropped_image,(new_image_size,new_image_size)))
        except:
            print("The cropped image is empty")
    
    return cropped_resized_images

data_images = []
data_labels = []

def process_image(file, data_dir, dataset):
    global data_images, data_labels
    file_name = file['file_name']
    img_annots = get_image_ground_truth(dataset, file_name)
    prop = get_image_proposals(data_dir + file_name)
    prop_categories = assign_category_to_proposal(prop, img_annots)
    cropped_resized_images = crop_images_to_proposals(data_dir + file_name, prop, new_image_size=cropped_image_size)
    cropped_resized_images_ground_truth = crop_images_to_proposals(data_dir + file_name,img_annots['bbox'], new_image_size=cropped_image_size)
    
    data_images = data_images + cropped_resized_images + cropped_resized_images_ground_truth
    data_labels = data_labels + prop_categories + img_annots['supercategory']

    """
    print(f"Processing {file}...")
    image = cv2.imread(data_dir+file)
    image = cv2.resize(image, (224, 224))
    
    boxes = np.array(selective_search(image, mode='single', random_sort=False), dtype='object')
    data_dir.split('/')[2]
    np.save(f"./data/proposals/{batch_idx}/{file.split('.')[0]}", boxes)
    """

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

    file_name = 'batch_1/000028.jpg'

    #data_dir = '/dtu/datasets1/02514/data_wastedetection/'
    data_dir = './data/'
    anns_file_path = data_dir + 'annotations.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    with Pool(processes=2) as pool:
        func = partial(process_image, data_dir=data_dir, dataset=dataset)
        pool.map(func, dataset['images'])

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    splits = sss.split(data_images, data_labels)
    x_train_samples, x_test_samples = [], []
    y_train_samples, y_test_samples = [], []
    for train_index, test_index in splits:
        x_train_samples.append(data_images[train_index]) 
        x_test_samples.append(data_images[test_index])
        y_train_samples.append(data_labels[train_index])
        y_test_samples.append(data_labels[test_index])


    """
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
    """
                