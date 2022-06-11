import joblib

sample = joblib.load('./data/split_dataset/train/train_image_1.pkl')
print(sample.shape)

with open('./data/split_dataset/train/train_image_1.pkl', 'rb') as pickle_file:
    sample = joblib.load(pickle_file)


print(sample.shape)