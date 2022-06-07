import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from data.data_prep import Hotdog_NotHotdog
from models.architectures import *

torch.cuda.empty_cache()

torch.manual_seed(42)
np.random.seed(77)

IMG_RESOLUTION = 128

CROSS_VALIDATION = True
K_SPLITS = 5

DATA_AUGMMENTATION = True

EPOCHS = 2
LR = 0.005
BATCH_SIZE = 64


def data_preparation():
    
    train_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor()])
    train_dataset = Hotdog_NotHotdog(train=True, transform=train_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')
    
    test_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor()])
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')

    if DATA_AUGMMENTATION:
        train_transforms_1 = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
                                            transforms.RandomRotation((90,90)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transforms_2 = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
                                                transforms.RandomRotation((180,180)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transforms_3 = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
                                                transforms.RandomRotation((270,270)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transforms_4 = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
                                                transforms.RandomHorizontalFlip(p=1.0),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transforms_5 = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
                                                transforms.RandomCrop(size=(IMG_RESOLUTION, IMG_RESOLUTION)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset_1 = Hotdog_NotHotdog(train=True, transform=train_transforms_1)
        trainset_2 = Hotdog_NotHotdog(train=True, transform=train_transforms_2)
        trainset_3 = Hotdog_NotHotdog(train=True, transform=train_transforms_3)
        trainset_4 = Hotdog_NotHotdog(train=True, transform=train_transforms_4)
        trainset_5 = Hotdog_NotHotdog(train=True, transform=train_transforms_5)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, trainset_1, trainset_2, trainset_3, trainset_4, trainset_5])
    
    return train_dataset, testset

def train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax):
    agg_labels, predictions = [], []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        model.train()
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output.to(device), labels.unsqueeze(1).to(torch.float32).to(device))

        loss.backward()
        optimizer.step()

        agg_labels.extend(labels.cpu().detach().numpy())
        if log_softmax:
            predictions.extend(torch.exp(output).cpu().detach().numpy())
        else:
            predictions.extend(output.cpu().detach().numpy())
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")

def validate(model, loss_func, val_loader, optimizer, device, log_softmax):
    predictions, agg_labels = [], []
    for batch_idx, (data, labels) in enumerate(val_loader):
        data, labels = data.to(device), labels.to(device)
        
        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        loss = loss_func(preds.to(device), labels.unsqueeze(1).to(torch.float32).to(device))

        if batch_idx % 20 == 0:
            print(f"Iteration {batch_idx}/{len(val_loader)}: Loss = {loss}")

        agg_labels.extend(labels.cpu().detach().numpy())
        if log_softmax:
            predictions.extend(torch.exp(preds).cpu().detach().numpy())
        else:
            predictions.extend(preds.cpu().detach().numpy())

    accuracy = performance_metrics(predictions, agg_labels, 'fold')
    return model, accuracy

def test(model, loss_func, test_loader, optimizer, device, log_softmax):
    predictions, agg_labels = [], []
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        
        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        save_samples(data, labels, preds)

        loss = loss_func(preds.to(device), labels.unsqueeze(1).to(torch.float32).to(device))

        if batch_idx % 20 == 0:
            print(f"Iteration {batch_idx}/{len(test_loader)}: Loss = {loss}")

        agg_labels.extend(labels.cpu().detach().numpy())
        if log_softmax:
            predictions.extend(torch.exp(preds).cpu().detach().numpy())
        else:
            predictions.extend(preds.cpu().detach().numpy())
    _ = performance_metrics(predictions, agg_labels, 'test')

def performance_metrics(predictions, labels, fold):
    outcome = np.argmax(predictions, axis=1)
    print(confusion_matrix(labels, outcome))
    test_accuracy =  np.sum([1 if item == labels[idx] else 0 for idx, item in enumerate(outcome)])/len(outcome)
    print(f"\n{fold} accuracy: {round(test_accuracy,3)}\n")
    return test_accuracy

def save_samples(data, labels, preds):
    pass


if __name__ == "__main__":

    train_dataset, testset = data_preparation()

    trainloaders_list, valloaders_list = [], []
    if CROSS_VALIDATION:
        kfold = KFold(n_splits=K_SPLITS, shuffle=False)
        for (train_ids, val_ids) in kfold.split(train_dataset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            trainloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=BATCH_SIZE,
                            sampler=train_subsampler,
                            num_workers=3))

            valloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=val_subsampler,
                            num_workers=3))

    else:
        trainloaders_list.append(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3))
        
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Running on {device}")

    loss_func = nn.BCELoss() #loss_func = nn.NLLLoss()
    if isinstance(loss_func, nn.BCELoss):
        log_softmax = False
    elif isinstance(loss_func, nn.NLLLoss):
        log_softmax = True
    
    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            model = Network().to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
        
            for epoch in range(1, EPOCHS):
                train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)
            
            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)]
    else:
        final_model = Network().to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=LR)
        for epoch in range(1, EPOCHS):
            train(final_model, loss_func, trainloaders_list, optimizer, epoch, device, log_softmax)

    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)

    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final.pt')
    torch.save(optimizer.state_dict(), './models/optim_final.pt')