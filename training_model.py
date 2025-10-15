# training phase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import random
import time
import copy
import os


# for i in range(2,4):
i = 5
tic = time.time()

# hyperparameters
n_epochs = 32
s_batch = 32
LR = 0.01
decay_epoch = 4
decay_ratio = 0.5
data_dir = f"./v622_level_5/"
phase_list = ['train', 'val']
reproducible = False
random_seed = 6   # works when reproducible is True

if(reproducible):
    seed = random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running device :', device)

data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize(240),
        #transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(240),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(240),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

datasets_ = { set_: datasets.ImageFolder(os.path.join(data_dir, set_), data_transforms[set_])
        for set_ in phase_list }
set_loader = { set_: DataLoader(datasets_[set_], batch_size=s_batch, shuffle=True, num_workers=0)
            for set_ in phase_list }
s_datasets = { set_: len(datasets_[set_]) for set_ in phase_list }
print(s_datasets)

# 選擇育訓練模型
class_list = datasets_['train'].classes
n_classes = len(class_list)
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
n_features = model.fc.in_features
#print(n_features)
model.fc = nn.Linear(n_features, n_classes)
model = model.to(device)

# 設定優化器(optimizer)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# 選擇 Loss Functions
criterion = nn.CrossEntropyLoss()

def lr_scheduler(optimizer, epoch, init_lr=LR, DE=decay_epoch, DR=decay_ratio):
    lr = init_lr*(DR**(epoch//DE))
    if(epoch%DE==0):
        print('設定 Learning Rate 為', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

selected_model = model
best_val_acc = 0
acc_records = {phase:[] for phase in phase_list}
model_name = ''
for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/{n_epochs}')
    print('-'*30)
    for phase in phase_list:
        tic_ = time.time()
        if(phase=='train'):
            model.train(True)
            optimizer = lr_scheduler(optimizer, epoch)
        else:
            model.train(False)

        running_loss = 0
        running_hits = 0

        for inputs, labels in set_loader[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            if(phase=='train'):
                loss.backward()
                optimizer.step()

            running_loss = running_loss + loss.data
            running_hits = running_hits + torch.sum(preds==labels.data).item()

        epoch_loss = running_loss/s_datasets[phase]
        epoch_acc = running_hits/s_datasets[phase]
        toc_ = time.time()
        print(f'{phase} --> Loss:{epoch_loss:.6f}  Acc:{epoch_acc*100:.6f}%  Elapsed_time:{toc_-tic_:.0f}s')
        acc_records[phase].append(round(epoch_acc,4))

        if(phase=='val' and epoch_acc>=best_val_acc):
            best_val_acc = epoch_acc
            selected_model = copy.deepcopy(model)
            # if(os.path.exists(model_name)):
            #     os.remove(model_name)
            acc_str = str(round(best_val_acc,5)).split('.')[1]
            model_name = f'v622_level{i}_r50_val_0' + acc_str + '.pkl'
            torch.save(selected_model, model_name)
    print('')
toc = time.time()

print(f'{model_name} completed in {round((toc-tic)/60)} minutes.')
del model
del optimizer
torch.cuda.empty_cache()
