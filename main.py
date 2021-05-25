import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
class CFG:
    seed = 54
    classes = 11014 
    scale = 30 
    margin = 0.5
    model_name =  'tf_efficientnet_b4'
    fc_dim = 512
    img_size = 512
    batch_size = 2
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '../input/utils-shopee/arcface_512x512_tf_efficientnet_b4_LR.pt'


# ----------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image, torch.tensor(1)


# ----------------------------------------------------------------------
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
        x = self.backbone(x)  #(2, 1792, 16, 16)
        #  print(x.shape)
        x = self.pooling(x).view(batch_size, -1)  #(2, 1792)
        #  print(x.shape)
        #  pdb.set_trace()

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x


# ----------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
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


# ----------------------------------------------------------------------
class DatasetPreparation(object):
    def __init__(self):
        pass

    def readDataCsv(self):
        self.df = pd.read_csv('./shopee-product-matching/train.csv')
        self.image_paths = './shopee-product-matching/train_images/' + self.df['image']

    def addMatchesGroundTruth(self):
        tmp = self.df.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df['matches'] = self.df['label_group'].map(tmp)
        self.df['matches'] = self.df['matches'].apply(lambda x: ' '.join(x))
        # print(self.df.head())

    def loadImageDataset(self):
        transforms = albumentations.Compose([
                albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)])
        self.image_dataset = ImageDataset(self.image_paths, transforms)





# ----------------------------------------------------------------------
if __name__ == '__main__':
    shopee_data = DatasetPreparation()
    shopee_data.readDataCsv()
    shopee_data.addMatchesGroundTruth()
    shopee_data.loadImageDataset()

    
    