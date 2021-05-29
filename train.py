import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import cudf
import cv2

import timm
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset,DataLoader

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import torchvision.transforms as transforms
import time
import pdb
MODEL_PATH = './model1/'


class CFG:
    seed = 54
    img_size = 512
    classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 15
    batch_size = 16
    num_workers = 8
    model_name = 'tf_efficientnet_b4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * batch_size,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output, nn.CrossEntropyLoss()(output,label)

class ShopeeModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.classes,
        model_name = CFG.model_name,
        fc_dim = CFG.fc_dim,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = True,
        pretrained = True):

        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        features = self.extract_features(image)
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x

def read_dataset():

    df = pd.read_csv('./shopee-product-matching/train.csv')
    df_cu = cudf.DataFrame(df)
    image_paths = './shopee-product-matching/train_images/' + df['image']

    return df, df_cu, image_paths

class ShopeeDataset(Dataset):

    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms
        self.labels = labels

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image, self.labels[index]

def get_train_transforms():
    return albumentations.Compose([
        albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
        albumentations.RandomCrop(CFG.img_size, CFG.img_size, always_apply=True),
        # albumentations.RandomHorizontalFlip(),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
    ])
    # transform_train = transforms.Compose([
    #     transforms.Resize(CFG.img_size),
    #     transforms.RandomCrop(CFG.img_size, padding=(CFG.img_size//8)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0,5,0,5,0,5),(0,5,0,5,0,5)),
    # ])
    return transform_train

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def train(model, trainloader, optimizer, scheduler, epoch, device):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, loss = model(inputs, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    scheduler.step()

def run_training():
    
    df,df_cu,image_paths = read_dataset()
    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])
    # pdb.set_trace()
    trainset = ShopeeDataset(image_paths=image_paths, labels=df['label_group'], transforms=get_train_transforms())
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = CFG.batch_size,
        num_workers = CFG.num_workers,
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    model = ShopeeModel()
    device_ids = [0,1,2,3]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(CFG.device)
    # pdb.set_trace()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = CFG.scheduler_params['lr_max'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.scheduler_params['lr_min'], T_max=CFG.epochs)
    
    for epoch in range(CFG.epochs):
        avg_loss_train = train(model, trainloader, optimizer, scheduler, epoch, CFG.device)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            MODEL_PATH + 'arcface_512x512_{}_checkpoints.pt'.format(CFG.model_name)
        )


run_training()