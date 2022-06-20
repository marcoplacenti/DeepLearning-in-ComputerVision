import os
import numpy as np
import glob
import PIL.Image as Image
from time import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim

#@title Data Loader
#data_path = 'isic_dataset/'
data_path = '/dtu/datasets1/02514/isic'
class ISIC(torch.utils.data.Dataset):
    def __init__(self, train, transform, seg = 0, data_path=data_path):
        'Initialization'
        self.transform = transform
        self.train = train
        data_path = os.path.join(data_path, 'train_allstyles' if train else 'test_style0')
        #self.image_paths = sorted(glob.glob(data_path + '/Images/*.jpg'))
        if train:
            if seg == 1:
                self.segmentation_paths = sorted(glob.glob(data_path + '/Segmentations/*_1_*.png'))
            elif seg == 2:
                self.segmentation_paths = sorted(glob.glob(data_path + '/Segmentations/*_2_*.png'))
            else:
                self.segmentation_paths = sorted(glob.glob(data_path + '/Segmentations/*_0_*.png'))
        else:
            self.segmentation_paths = sorted(glob.glob(data_path + '/Segmentations/*.png'))

        names = [seg.split("/")[-1].split("_")[1] for seg in self.segmentation_paths]
        self.image_paths = [glob.glob(data_path + '/Images/ISIC_' + filename + '.jpg')[0] for filename in names]            
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        X = self.transform(image)
        
        segmentation_path = self.segmentation_paths[idx]
        segmentation = Image.open(segmentation_path)   
        Y = self.transform(segmentation)
        
        #if not self.train:
        #    Y = Y[1,:,:]
        
        return X, Y

#@title Basic UNet model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor = 2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor = 2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor = 2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor = 2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64 + 64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = self.pool0(e0)     
        e1 = F.relu(self.enc_conv1(e1))
        e2 = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e2))
        e3 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e3))
        
        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = self.upsample0(b)
        d0 = self.dec_conv0(torch.cat([d0, e3],1))
        d1 = self.upsample1(d0)
        d2 = self.dec_conv1(torch.cat([d1, e2], 1))
        d2 = self.upsample2(d2)
        d3 = self.dec_conv2(torch.cat([d2, e1], 1))
        d3 = self.upsample3(d3) 
        d3 = self.dec_conv3(torch.cat([d3, e0], 1))

        return d3


#@title Improved UNet model with dilation
class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.Conv2d(64, 64, 3, 2, padding=1, dilation = 1) # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.Conv2d(64, 64, 3, 2, padding=2, dilation = 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, 3, 2, padding=3, dilation = 3)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, 3, 2, padding=4, dilation = 4)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(64, 64, 3, 2, padding = 1, output_padding=1, dilation = 1)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 3, 2, padding = 2, output_padding=1, dilation = 2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 64, 3, 2, padding = 3, output_padding=1, dilation = 3)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 64, 3, 2, padding = 4, output_padding=1, dilation = 4)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64 + 64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = self.pool0(e0)  
        e1 = F.relu(self.enc_conv1(e1))
        e2 = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e2))
        e3 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e3))
        
        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = self.upsample0(b)
        d0 = self.dec_conv0(torch.cat([d0, e3],1))
        d1 = self.upsample1(d0)
        d2 = self.dec_conv1(torch.cat([d1, e2], 1))
        d2 = self.upsample2(d2)
        d3 = self.dec_conv2(torch.cat([d2, e1], 1))
        d3 = self.upsample3(d3) 
        d3 = self.dec_conv3(torch.cat([d3, e0], 1))

        return d3



#@title Loaders
def simple_data_load(batch_size = 32, size = 128):
  train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])

  test_transform = transforms.Compose([transforms.CenterCrop((215, 205)), transforms.Resize((size, size)), 
                                      transforms.ToTensor()])

  trainset0 = ISIC(train=True, transform=train_transform, seg = 0)
  trainset1 = ISIC(train=True, transform=train_transform, seg = 1)
  trainset2 = ISIC(train=True, transform=train_transform, seg = 2)
  trainset = torch.utils.data.ConcatDataset([trainset0, trainset1, trainset2])

  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testset = ISIC(train=False, transform=test_transform)
  test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

  print('Loaded %d training images' % len(trainset))
  print('Loaded %d test images' % len(testset))
  return train_loader, test_loader

def augmented_data_load(batch_size = 32, size = 128):
    train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])

    augment1_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment2_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1.0),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment3_transform = transforms.Compose([transforms.RandomRotation((90,90)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment4_transform = transforms.Compose([transforms.RandomRotation((-90,-90)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment5_transform = transforms.Compose([transforms.RandomRotation((180,180)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment6_transform = transforms.Compose([transforms.RandomRotation((-180,-180)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment7_transform = transforms.Compose([transforms.GaussianBlur(3, 0.2),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])
        
    test_transform = transforms.Compose([transforms.CenterCrop((200, 200)), transforms.Resize((size, size)), 
                                        transforms.ToTensor()])
    
    trainset0 = ISIC(train=True, transform=train_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=train_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=train_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment1_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment1_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment1_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment2_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment2_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment2_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment3_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment3_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment3_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment4_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment4_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment4_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment5_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment5_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment5_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment6_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment6_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment6_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    trainset0 = ISIC(train=True, transform=augment7_transform, seg = 0)
    trainset1 = ISIC(train=True, transform=augment7_transform, seg = 1)
    trainset2 = ISIC(train=True, transform=augment7_transform, seg = 2)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0, trainset1, trainset2])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = ISIC(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print('Loaded %d training images' % len(trainset))
    print('Loaded %d test images' % len(testset))
    return train_loader, test_loader

    
def augmented_data_load_single_style(batch_size = 32, size = 128, style = 0):
    train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])

    augment1_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment2_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1.0),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment3_transform = transforms.Compose([transforms.RandomRotation((90,90)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment4_transform = transforms.Compose([transforms.RandomRotation((-90,-90)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment5_transform = transforms.Compose([transforms.RandomRotation((180,180)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment6_transform = transforms.Compose([transforms.RandomRotation((-180,-180)),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

    augment7_transform = transforms.Compose([transforms.GaussianBlur(3, 0.2),
                                            transforms.Resize((size, size)), 
                                            transforms.ToTensor()])
        
    test_transform = transforms.Compose([transforms.CenterCrop((200, 200)), transforms.Resize((size, size)), 
                                        transforms.ToTensor()])
    


    trainset = ISIC(train=True, transform=train_transform, seg = style)

    trainset0 = ISIC(train=True, transform=augment1_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment2_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment3_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment4_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment5_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment6_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    trainset0 = ISIC(train=True, transform=augment7_transform, seg = style)
    trainset = torch.utils.data.ConcatDataset([trainset, trainset0])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = ISIC(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print('Loaded %d training images' % len(trainset))
    print('Loaded %d test images' % len(testset))
    return train_loader, test_loader

#@title Training, Loss and Performance Evaluation
def dice_score(pred, target):
    pred = pred.long()
    target = target.long()
    smooth = 1.
    #m1 = torch.reshape(pred, [-1])
    #m2 = torch.reshape(target, [-1])
    intersection = torch.mul(pred, target).sum().float()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    torch.cuda.empty_cache()
    X_test, Y_test = next(iter(test_loader))
    
    train_perf = []
    test_perf = []
    train_loss = []
    test_loss = []
    
    for epoch in range(epochs):
        tic = time()
        #print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        avg_perf = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = nn.Sigmoid()(model(X_batch))
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            perf = dice_score(Y_pred >= 0.5, Y_batch >= 0.5)
            #print(loss, perf)
            loss.backward()  # backward-pass
            opt.step()  # update weights
            avg_loss += loss
            avg_perf += perf

        # calculate metrics to show the user
        avg_loss = avg_loss / len(train_loader)
        avg_perf = avg_perf / len(train_loader) #?
        print(f"Loss: {avg_loss.item()}, Perf: {avg_perf.item()}")
        toc = time()
        #print(' - loss: %f' % avg_loss)
                
        train_loss.append(avg_loss.detach().cpu())
        train_perf.append(avg_perf.detach().cpu())
        # show intermediate results
        avg_loss = 0
        avg_perf = 0

        model.eval()  # testing mode
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            #Y_batch = Y_batch[:,1,:,:].to(device)
            Y_hat = nn.Sigmoid()(model(X_batch))#.squeeze(1)#[:,0,:,:]
            loss = loss_fn(Y_hat, Y_batch)
            perf = dice_score(Y_hat >= 0.5, Y_batch)
            #print(loss, perf)
            avg_loss += loss
            avg_perf += perf
        
        # calculate metrics to show the user
        avg_loss = avg_loss / len(train_loader)
        avg_perf = avg_perf / len(train_loader) #?
        print(f"Loss: {avg_loss}, Perf: {avg_perf}")
        test_loss.append(avg_loss.detach().cpu())
        test_perf.append(avg_perf.detach().cpu())
            
        Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        #clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0] >= 0.5)#, cmap='gray')
            #plt.imshow(Y_hat[k, 0])#, cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        #plt.savefig("figures/image_name.png")
    return [train_loss, test_loss, train_perf, test_perf]

def confusion_mat(model, test_loader):
    TP, TN, FP, FN = 0,0,0,0
    for images, labels in test_loader:
        Y_pred = torch.reshape(torch.sigmoid(model(images.to(device))), [-1]).detach().cpu().numpy() >= 0.5
        Y_target = torch.reshape(labels[:,1,:,:].unsqueeze(1), [-1]).detach().cpu().numpy() >= 0.5
        TP += np.sum((Y_pred == 1) & (Y_target == 1)) / len(Y_pred)
        TN += np.sum((Y_pred == 0) & (Y_target == 0)) / len(Y_pred)
        FP += np.sum((Y_pred == 1) & (Y_target == 0)) / len(Y_pred)
        FN += np.sum((Y_pred == 0) & (Y_target == 1)) / len(Y_pred)
        
    return TP, TN, FP, FN


def print_model_performance(model, test_loader, performance):
    plt.rcParams['figure.figsize'] = [18, 6]

    images, labels = next(iter(test_loader))
    predicted_segmentations = torch.sigmoid(model(images.to(device))).detach().cpu()

    train_loss = performance[0]
    test_loss = performance[1]
    train_perf = performance[2]
    test_perf = performance[3]
    for i in range(6):
        plt.subplot(4, 6, i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))
        if i == 0: plt.ylabel('Images')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 6, i+7)
        plt.imshow(labels[i][1])
        if i == 0: plt.ylabel('Segmentations')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 6, i+13)
        plt.imshow(predicted_segmentations[i, 0] >= 0.5)
        if i == 0: plt.ylabel('Model')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 6, i+19)
        plt.imshow(predicted_segmentations[i, 0])
        if i == 0: plt.ylabel('Model')
        plt.xticks([])
        plt.yticks([])

    #plt.show()

    plt.plot(train_loss, '--')
    plt.plot(test_loss, '--')
    plt.plot(train_perf, '--')
    plt.plot(test_perf, '--')
    plt.legend(['Train loss', 'Test loss','Train performance', 'Test performance'])
    plt.ylabel('Performance')
    plt.xlabel('Epochs')
    plt.xticks(range(1,21))
    plt.grid()
    #plt.show()

    print(confusion_mat(model, test_loader))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Device count: " + str(torch.cuda.device_count()))

    train_loader, test_loader = simple_data_load()

    """
    model = UNet().to(device)
    summary(model, (3, 128, 128))
    performance = train(model, optim.Adam(model.parameters(), 0.0001), nn.BCELoss(), 20, train_loader, test_loader)
    print_model_performance(model, test_loader, performance)
    
    train_loader, test_loader = augmented_data_load(batch_size = 16)
    #if 'model' in locals(): del model
    torch.cuda.empty_cache()
    model = UNet().to(device)
    summary(model, (3, 128, 128))
    performance = train(model, optim.Adam(model.parameters(), 0.0001), nn.BCELoss(), 20, train_loader, test_loader)
    print_model_performance(model, test_loader, performance)
    """
    #train_loader, test_loader = augmented_data_load(batch_size = 32)
    #if 'model' in locals(): del model
    torch.cuda.empty_cache()
    model = DilatedNet().to(device)
    summary(model, (3, 128, 128))
    performance = train(model, optim.Adam(model.parameters(), 0.0001), nn.BCELoss(), 10, train_loader, test_loader)
    print_model_performance(model, test_loader, performance)
