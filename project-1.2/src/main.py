import numpy as np
import os
import json
from copy import deepcopy
import joblib

from multiprocessing import Pool, Process, Manager
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

def get_image_proposals(filepath,img_annots):
    # Get the proposals for an image
    image = cv2.imread(filepath)
    image = cv2.resize(image,(img_annots['new_size'][0],img_annots['new_size'][1]))
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
    prop_filtered = []

    for i in range(len(prop)):
        if not (assignemt_prob[i,:] >= 0.7).any():
            if (assignemt_prob[i, :] < 0.3).any():
                prop_filtered.append(prop[i])
                prop_categories.append('background')
        else:
            index = np.argmax(assignemt_prob[i,:])
            prop_categories.append(img_annots['supercategory'][index])
            prop_filtered.append(prop[i])
            
    return prop_categories, prop_filtered

def crop_images_to_proposals(filepath, prop, new_image_size, img_annots):
    image = cv2.imread(filepath)
    image = cv2.resize(image,(img_annots['new_size'][0],img_annots['new_size'][1]))
    cropped_resized_images = []
    for box in prop:
        cropped_image = image[int(box[1]):int(box[1]+box[3]),int(box[0]):int(box[0] + box[2])]
        try:
            cropped_resized_images.append(cv2.resize(cropped_image,(new_image_size,new_image_size)))
        except:
            print("The cropped image is empty")
    
    return cropped_resized_images
        

def process_image(file, data_dir, dataset):
    file_name = file['file_name']
    print(file_name)
    img_annots = get_image_ground_truth(dataset, file_name)
    prop = get_image_proposals(data_dir + file_name, img_annots)
    prop_categories, prop_filtered = assign_category_to_proposal(prop, img_annots)
    cropped_resized_images = crop_images_to_proposals(data_dir + file_name, prop_filtered, cropped_image_size, img_annots)
    cropped_resized_images_ground_truth = crop_images_to_proposals(data_dir + file_name,img_annots['bbox'], cropped_image_size, img_annots)
    
    img = cropped_resized_images + cropped_resized_images_ground_truth
    label = prop_categories + img_annots['supercategory']

    return img, label, file_name


if __name__ == '__main__':

    data_dir = '/dtu/datasets1/02514/data_wastedetection/'
    #data_dir = './data/'
    anns_file_path = data_dir + 'annotations.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    dataset_size = len(dataset['images'])
    train_dataset_size = int(0.8*dataset_size)
    validation_dataset_size = int(0.1*dataset_size)
    test_dataset_size = dataset_size - (train_dataset_size + validation_dataset_size)

    np.random.seed(42)
    arr = np.arange(0,dataset_size)
    np.random.shuffle(arr)

    train_dataset_id = arr[0:train_dataset_size]
    validation_dataset_id = arr[train_dataset_size:train_dataset_size+validation_dataset_size]
    test_dataset_id = arr[train_dataset_size+validation_dataset_size:train_dataset_size+validation_dataset_size+test_dataset_size]

    train_set, val_set, test_set = [], [], []
    for m in range(dataset_size):
        candidate = dataset['images'][m]
        if candidate['id'] in train_dataset_id:
            train_set.append(candidate)
        elif candidate['id'] in validation_dataset_id:
            val_set.append(candidate)
        elif candidate['id'] in test_dataset_id:
            test_set.append(candidate)

    if not os.path.exists('./data/split_dataset/train'):
        os.makedirs('./data/split_dataset/train')
    if not os.path.exists('./data/split_dataset/val'):
        os.makedirs('./data/split_dataset/val')
    if not os.path.exists('./data/split_dataset/test'):
        os.makedirs('./data/split_dataset/test')
    
    """
    print("Processing validation...")
    with Pool(processes=4) as pool:
        func = partial(process_image, data_dir=data_dir, dataset=dataset)
        vals = pool.map(func, val_set)
        images = [pair[0] for pair in vals]
        labels = [pair[1] for pair in vals]
        file_names = [pair[2] for pair in vals]
    images = np.array(images, dtype='object')
    for idx, image in enumerate(images):
        joblib.dump(image, f'./data/split_dataset/val/val_images_{idx}.pkl', compress=5)
    #np.save('./data/split_dataset/val_images.npy', images)

    labels = np.array(labels, dtype='object')
    joblib.dump(labels, './data/split_dataset/val/val_labels.pkl', compress=5)
    #np.save('./data/split_dataset/val_labels.npy', labels)

    joblib.dump(np.array(file_names, './data/split_dataset/val_filenames.pkl'), compress=5)

    print("Processing testing...")
    with Pool(processes=4) as pool:
        func = partial(process_image, data_dir=data_dir, dataset=dataset)
        vals = pool.map(func, test_set)
        images = [pair[0] for pair in vals]
        labels = [pair[1] for pair in vals]
        file_names = [pair[2] for pair in vals]
    images = np.array(images, dtype='object')
    for idx, image in enumerate(images):
        joblib.dump(image, f'./data/split_dataset/test/test_images_{idx}.pkl', compress=5)
    #np.save('./data/split_dataset/test_images.npy', images)

    labels = np.array(labels, dtype='object')
    joblib.dump(labels, './data/split_dataset/test/test_labels.pkl', compress=5)
    #np.save('./data/split_dataset/test_labels.npy', labels)

    joblib.dump(np.array(file_names, './data/split_dataset/test_filenames.pkl'), compress=5)
    """
    print("Processing training...")
    with Pool(processes=4) as pool:
        func = partial(process_image, data_dir=data_dir, dataset=dataset)
        vals = pool.map(func, train_set)
        images = [pair[0] for pair in vals]
        labels = [pair[1] for pair in vals]
        file_names = [pair[2] for pair in vals]
    images = np.array(images, dtype='object')
    for idx, image in enumerate(images):
        image = np.array(image, dtype='object')
        print(idx, image.shape)
        joblib.dump(image, f'./data/split_dataset/train/train_images_{idx}.pkl', compress=5)
    #np.save('./data/split_dataset/train_images.npy', images)

    labels = np.array(labels, dtype='object')
    joblib.dump(labels, './data/split_dataset/train/train_labels.pkl', compress=5)
    #np.save('./data/split_dataset/train_labels.npy', labels)

    joblib.dump(np.array(file_names, './data/split_dataset/train_filenames.pkl'), compress=5)
                