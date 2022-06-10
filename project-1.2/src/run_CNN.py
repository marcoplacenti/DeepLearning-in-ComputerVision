import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from data.data_prep import Hotdog_NotHotdog
from models.architectures import Network, VGG
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim

torch.cuda.empty_cache()
IMG_RESOLUTION = 224
EPOCHS = 19
LR = 0.005
BATCH_SIZE = 64

def data_preparation():
    # load data here and 

    train_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #train_dataset = Hotdog_NotHotdog(train=True, transform=train_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')
    
    test_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #testset = Hotdog_NotHotdog(train=False, transform=test_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')

    pass


def train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=10):
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
            
            target = target.to(torch.float32)
            #print(len(target))
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            #with torch.no_grad():
            output = model(data)
            
            #Compute the loss
            loss = loss_func(output, target.reshape(-1,1))
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = torch.flatten(torch.where(output > 0.5, 1.0, 0.0))
            train_correct += (target==predicted).sum().cpu().item()
            #print(len(output))
            #print(len(predicted))
            #print("------------------")
            
        #print(train_correct)
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.to(torch.float32)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_func(output, target.reshape(-1,1)).cpu().item())
            predicted = torch.flatten(torch.where(output > 0.5, 1.0, 0.0))
            test_correct += (target==predicted).sum().cpu().item()
            
        out_dict['train_acc'].append(train_correct/len(train_dataset_conc))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict

def test(model, loss_func, test_loader, optimizer, device, log_softmax):
    predictions, agg_labels = [], []
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)

        if not log_softmax:
            labels = labels.reshape(-1,1).to(torch.float32).to(device)
        
        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        save_samples(data, labels, preds)

        loss = loss_func(preds, labels)

        if batch_idx % 20 == 0:
            print(f"Iteration {batch_idx}/{len(test_loader)}: Loss = {loss}")

        agg_labels.extend(labels.cpu().detach().numpy())
        if log_softmax:
            predictions.extend(torch.exp(preds).cpu().detach().numpy())
        else:
            predictions.extend(preds.cpu().detach().numpy())


if __name__ == "__main__":

    train_dataset,validation_set, testset = data_preparation()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Running on {device}")

    loss_func = nn.NLLLoss()
    log_softmax = True 

    final_model = models.resnet152(pretrained=True)
    num_ftrs = final_model.fc.in_features
    final_model.fc = nn.Sequential(nn.Linear(num_ftrs, 29),nn.Softmax())
    final_model = final_model.to(device)
    optimizer = optim.Adam(final_model.parameters())
    out_dict_res = train2(final_model, loss_func, train_loader, validation_loader, optimizer, device, log_softmax, num_epochs=10)
    
    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)    


    



