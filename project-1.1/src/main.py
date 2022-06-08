import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
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

#torch.manual_seed(42)
#np.random.seed(77)

IMG_RESOLUTION = 128

CROSS_VALIDATION = True
K_SPLITS = 5

DATA_AUGMENTATION = True

EPOCHS = 19
LR = 0.005
BATCH_SIZE = 64


def data_preparation():
    
    train_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = Hotdog_NotHotdog(train=True, transform=train_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')
    
    test_transform = transforms.Compose([transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)#, data_path='dtu/datasets1/02514/hotdog_nothotdog/')

    if DATA_AUGMENTATION:
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
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, labels = data.to(device), labels.to(device)

        if not log_softmax:
            labels = labels.reshape(-1,1).to(torch.float32).to(device)

        model.train()
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, labels)

        loss.backward()
        optimizer.step()

        agg_labels.extend(labels.cpu().detach().numpy())
        if log_softmax:
            predictions.extend(torch.exp(output).cpu().detach().numpy())
        else:
            predictions.extend(output.cpu().detach().numpy())
        #if batch_idx % 20 == 0:
        #    print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss}")

    _ = performance_metrics(predictions, agg_labels, 'train')

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

def validate(model, loss_func, val_loader, optimizer, device, log_softmax):
    predictions, agg_labels = [], []
    for batch_idx, (data, labels) in enumerate(val_loader):
        data, labels = data.to(device), labels.to(device)

        if not log_softmax:
            labels = labels.reshape(-1,1).to(torch.float32).to(device)
        
        model.eval()
        optimizer.zero_grad()
        preds = model(data)

        loss = loss_func(preds, labels)

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

    #print(predictions)
    #print(agg_labels)
    _ = performance_metrics(predictions, agg_labels, 'test')

def performance_metrics(predictions, labels, fold):
    outcome = np.argmax(predictions,axis=1)
    print(confusion_matrix(labels, outcome))
    test_accuracy =  np.sum([1 if item == labels[idx] else 0 for idx, item in enumerate(outcome)])/len(outcome)
    print(f"\n{fold} accuracy: {round(test_accuracy,3)}\n")
    return test_accuracy

def save_samples(data, labels, preds):
    pass


if __name__ == "__main__":

    train_dataset_conc, testset = data_preparation()

    trainloaders_list, valloaders_list = [], []
    if CROSS_VALIDATION:
        kfold = KFold(n_splits=K_SPLITS, shuffle=True)
        for (train_ids, val_ids) in kfold.split(train_dataset_conc):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            trainloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset_conc, 
                            batch_size=BATCH_SIZE,
                            #shuffle=True,
                            sampler=train_subsampler,
                            num_workers=3))

            valloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset_conc,
                            batch_size=BATCH_SIZE,
                            #shuffle=True,
                            sampler=val_subsampler,
                            num_workers=3))

    else:
        #train_loader = DataLoader(train_dataset_conc, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        trainloaders_list.append(DataLoader(train_dataset_conc, batch_size=BATCH_SIZE, shuffle=True, num_workers=3))
        
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Running on {device}")

    loss_func = nn.BCELoss() 
    #loss_func = nn.NLLLoss()
    if isinstance(loss_func, nn.BCELoss):
        log_softmax = False
    elif isinstance(loss_func, nn.NLLLoss):
        log_softmax = True
    
    #-------------------------------------------------
    # Network with ADAM and data aug
    #-------------------------------------------------

    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = Network().to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            out_dict = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=20)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)
            
            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = Network().to(device)
        optimizer = optim.Adam(final_model.parameters())
        out_dict = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=20)
        with open('adam_aug_128.json', 'w') as fp:
            json.dump(out_dict, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)
    # TODO - save values from the last test run
    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)
    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_adam_aug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_adam_aug_128.pt')

    # -------------------------------------------------
    # Network with SGD and data aug
    #---------------------------------------------------
    final_model = None

    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = Network().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.90)
            out_dict_sgd = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=20)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)

            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = Network().to(device)
        optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.90)
        out_dict_sgd = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=20)
        with open('sgd_aug_128.json', 'w') as fp:
            json.dump(out_dict_sgd, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)

    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)

    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_sgd_aug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_sgd_aug_128.pt')

    #---------------------------------------------------
    # Network with Resnet152 and data aug
    #---------------------------------------------------

    final_model = None

    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.90)
            out_dict_res = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=10)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)

            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = models.resnet152(pretrained=True)
        num_ftrs = final_model.fc.in_features
        final_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())
        final_model = final_model.to(device)
        optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.90)
        out_dict_res = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=10)
        with open('res_aug_128.json', 'w') as fp:
            json.dump(out_dict_res, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)

    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)

    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_res_aug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_res_aug_128.pt')

    #---------------------------------------------------
    # Network without data augmentation with adam
    #---------------------------------------------------
    final_model = None
    DATA_AUGMENTATION = False

    train_dataset_conc, testset = data_preparation()
    trainloaders_list, valloaders_list = [], []
    if CROSS_VALIDATION:
        kfold = KFold(n_splits=K_SPLITS, shuffle=False)
        for (train_ids, val_ids) in kfold.split(train_dataset_conc):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            trainloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset_conc,
                            batch_size=BATCH_SIZE,
                            #shuffle=True,
                            sampler=train_subsampler,
                            num_workers=3))

            valloaders_list.append(torch.utils.data.DataLoader(
                            train_dataset_conc,
                            batch_size=BATCH_SIZE,
                            #shuffle=True,
                            sampler=val_subsampler,
                            num_workers=3))

    else:
        trainloaders_list.append(DataLoader(train_dataset_conc, batch_size=BATCH_SIZE, shuffle=True, num_workers=3))

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = Network().to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            out_dict = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=20)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)

            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = Network().to(device)
        optimizer = optim.Adam(final_model.parameters())
        out_dict = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=20)
        with open('adam_noaug_128.json', 'w') as fp:
            json.dump(out_dict, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)
    # TODO - save values from the last test run
    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)
    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_adam_noaug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_adam_noaug_128.pt')

 
    #---------------------------------------------------
    # Network without data augmentation with sgd
    #---------------------------------------------------

    final_model = None
    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = Network().to(device)
            optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.90)
            out_dict_sgd = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=20)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)

            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = Network().to(device)
        optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.90)
        out_dict_sgd = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=20)
        with open('sgd_noaug_128.json', 'w') as fp:
            json.dump(out_dict_sgd, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)

    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)

    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_sgd_noaug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_sgd_noaug_128.pt')

    #----------------------------------------------------
    #----------------------------------------------------
    final_model = None

    if CROSS_VALIDATION:
        models_accuracies = {}
        for i, train_loader in enumerate(trainloaders_list):
            #model = VGG(3, 2).to(device)
            model = models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.90)
            out_dict_res = train2(model, loss_func, train_loader, test_loader, optimizer, device, log_softmax, num_epochs=10)
            #for epoch in range(1, EPOCHS):
            #    train(model, loss_func, train_loader, optimizer, epoch, device, log_softmax)

            kf_model, accuracy = validate(model, loss_func, valloaders_list[i], optimizer, device, log_softmax)
            models_accuracies[accuracy] = kf_model

            final_model = [models_accuracies[key] for key in sorted(models_accuracies.keys(), reverse=True)][0]

    else:
        #final_model = VGG(3, 2).to(device)
        final_model = models.resnet152(pretrained=True)
        num_ftrs = final_model.fc.in_features
        final_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1),nn.Sigmoid())
        final_model = final_model.to(device)
        optimizer = optim.SGD(final_model.parameters(), lr=0.01, momentum=0.90)
        out_dict_res = train2(final_model, loss_func, trainloaders_list[0], test_loader, optimizer, device, log_softmax, num_epochs=10)
        with open('res_noaug_128.json', 'w') as fp:
            json.dump(out_dict_res, fp)
        #for epoch in range(1, EPOCHS):
        #    train(final_model, loss_func, trainloaders_list[0], optimizer, epoch, device, log_softmax)

    test(final_model, loss_func, test_loader, optimizer, device, log_softmax)

    print("Saving model weights and optimizer...")
    torch.save(final_model.state_dict(), './models/model_final_res_noaug_128.pt')
    torch.save(optimizer.state_dict(), './models/optim_final_res_noaug_128.pt')


