import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from data.data_prep import Hotdog_NotHotdog
from models.architectures import *


def data_preparation():
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

    return train_loader, test_loader

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




if __name__ == "__main__":

    train_loader, test_loader = data_preparation()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    loss_func = nn.NLLLoss()

    for epoch in range(1, 5+1):
        train(model, loss_func, train_loader, optimizer, epoch, device)

    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)

        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        loss = loss_func(preds, labels)

        print(f"Iteration {batch_idx}/{len(test_loader)}: Loss = {loss}")


    print("Saving model weights and optimizer...")
    torch.save(model.state_dict(), 'src/models/model_final.pt')
    torch.save(optimizer.state_dict(), 'src/models/optim_final.pt')