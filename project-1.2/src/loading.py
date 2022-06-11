import joblib
import os
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader


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

tensor_x = torch.Tensor(val_data) # transform to torch tensor
tensor_y = torch.Tensor(val_labels)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader

print("done")
exit()

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