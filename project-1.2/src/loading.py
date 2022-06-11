import joblib
import os
import numpy as np
from tqdm import tqdm

imgs = os.listdir('./data/split_dataset/val/')

val_data_sep = []
val_data = []
for idx, img in tqdm(enumerate(imgs)):
    if img.startswith('val_image'):
        img_proposals = joblib.load('./data/split_dataset/val/'+img)
        val_data_sep.append(img_proposals)
        val_data.extend(img_proposals)

val_labels = joblib.load('./data/split_dataset/val/val_labels.pkl')

val_filenames = joblib.load('./data/split_dataset/val_filenames.pkl')

print("done")


imgs = os.listdir('./data/split_dataset/test/')

val_data_sep = []
val_data = []
for idx, img in tqdm(enumerate(imgs)):
    if img.startswith('test_image'):
        img_proposals = joblib.load('./data/split_dataset/test/'+img)
        val_data_sep.append(img_proposals)
        val_data.extend(img_proposals)

val_labels = joblib.load('./data/split_dataset/test/test_labels.pkl')

val_filenames = joblib.load('./data/split_dataset/test_filenames.pkl')

print("done")


imgs = os.listdir('./data/split_dataset/train/')

val_data_sep = []
val_data = []
for idx, img in tqdm(enumerate(imgs)):
    if img.startswith('train_image'):
        img_proposals = joblib.load('./data/split_dataset/train/'+img)
        val_data_sep.append(img_proposals)
        val_data.extend(img_proposals)

val_labels = joblib.load('./data/split_dataset/train/train_labels.pkl')

val_filenames = joblib.load('./data/split_dataset/train_filenames.pkl')

print("done")