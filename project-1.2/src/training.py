from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
from tqdm import tqdm

def train(model, optimizer, train_loader, validation_loader, num_epochs):
    def loss_fun(output, target):
        return F.cross_entropy(output, target)
        #return F.binary_cross_entropy(output, target)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #print(output)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)

            train_correct += (target==predicted).sum().cpu().item()

            
        #print(train_correct)
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            
        out_dict['train_acc'].append(train_correct/len(train_targets))
        out_dict['test_acc'].append(test_correct/len(validation_targets))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")

    return out_dict

from torchvision import models

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 29))
model_ft = model_ft.to(device)

optimizer = torch.optim.Adam(model_ft.parameters())

train_loader = torch.load('./data/split_dataset/train/train_data_loader.pth')
validation_loader = torch.load('./data/split_dataset/val/val_data_loader.pth')
test_loader = torch.load('./data/split_dataset/test/test_data_loader.pth')


out_dict_resnet = train(model_ft, optimizer, train_loader, validation_loader, 3)

def loss_fun(output, target):
    return F.cross_entropy(output, target)
model_ft.eval()
test_loss = []
test_correct = 0
targets = np.array([])
predictions = np.array([])
scores = np.array([])

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model_ft(data)
    test_loss.append(loss_fun(output, target).cpu().item())
    predicted = output.argmax(1)
    scor = torch.max(output, 1)[0]
    test_correct += (target==predicted).sum().cpu().item()
    targets = np.concatenate((targets,target.cpu().detach().numpy()))
    predictions = np.concatenate((predictions,predicted.cpu().detach().numpy()))
    scores = np.concatenate((scores,scor.cpu().detach().numpy()))
    