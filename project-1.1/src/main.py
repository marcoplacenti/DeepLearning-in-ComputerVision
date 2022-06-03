import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from data.data_prep import Hotdog_NotHotdog



if __name__ == "__main__":
    size = 128
    train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(len(train_loader))