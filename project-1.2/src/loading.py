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

print(np.array(val_data_sep).shape)
print(np.array(val_data).shape)
print(np.array(val_labels).shape)
print(np.array(val_filenames).shape)