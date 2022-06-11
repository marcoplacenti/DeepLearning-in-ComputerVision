import pickle

with open('./data/split_dataset/train/train_image_1.pkl', 'rb') as pickle_file:
    sample = pickle.load(pickle_file)
#sample = pickle.load('./data/split_dataset/train/train_image_1.pkl')

print(sample.shape)