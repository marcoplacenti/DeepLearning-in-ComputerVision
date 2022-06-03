import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix

from data.data_prep import Hotdog_NotHotdog
from models.architectures import *


def data_preparation():
    size = 128
    train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return trainset, train_loader, testset, test_loader

def train(model, loss_func, train_loader, optimizer, epoch, device):
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        model.train()
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, labels)

        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")

def performance_metrics(predictions, labels):
    outcome = np.argmax(predictions, axis=1)

    print(confusion_matrix(labels, outcome))

    test_accuracy =  np.sum([1 if item == labels[idx] else 0 for idx, item in enumerate(outcome)])/len(outcome)



    print(f"\nTest Accuracy: {round(test_accuracy,3)}\n")


if __name__ == "__main__":

    trainset, train_loader, testset, test_loader = data_preparation()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Running on {device}")
    
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    loss_func = nn.NLLLoss()

    for epoch in range(1, 5+1):
        train(model, loss_func, train_loader, optimizer, epoch, device)

    predictions, agg_labels = [], []
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        agg_labels.extend(labels)

        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        loss = loss_func(preds, labels)

        if batch_idx % 20 == 0:
            print(f"Iteration {batch_idx}/{len(test_loader)}: Loss = {loss}")

        predictions.extend(torch.exp(preds).cpu().detach().numpy())
    
    performance_metrics(predictions, agg_labels)

    print("Saving model weights and optimizer...")
    torch.save(model.state_dict(), './models/model_final.pt')
    torch.save(optimizer.state_dict(), './models/optim_final.pt')