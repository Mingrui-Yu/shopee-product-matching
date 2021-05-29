import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

import gc
import pdb
from collections import OrderedDict
import utils

# ----------------------------------------------------------------------
class Params:
    classes = 11014 
    margin = 0.5
    model_name =  'tf_efficientnet_b4'
    fc_dim = 512
    scale = 30 
    batch_size = 24
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './model1/arcface_512x512_tf_efficientnet_b4_checkpoints.pt'
    need_calc_embeddings = True



# ----------------------------------------------------------------------
class ShopeeModel(nn.Module):
    def __init__(
            self,
            n_classes = Params.classes,
            model_name = Params.model_name,
            fc_dim = Params.fc_dim,
            margin = Params.margin,
            scale = Params.scale,
            use_fc = True,
            pretrained = True,
            b_training = True):

        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        self.training = b_training

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
class MatchingNN(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def getImageEmbeddings(self):
        # model = ShopeeModel(pretrained=True).to(Params.device)

        model = ShopeeModel(pretrained=False)
        checkpoint = torch.load(Params.model_path, map_location='cpu')

        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(Params.device)

        model.eval()

        image_loader = torch.utils.data.DataLoader(
            self.dataset.image_dataset,
            batch_size=Params.batch_size,
            num_workers=Params.num_workers
        )

        embeds = []
        with torch.no_grad():
            for img, label in tqdm(image_loader): 
                img = img.cuda()
                label = label.cuda()
                features = model(img, label)
                image_embeddings = features.detach().cpu().numpy()
                embeds.append(image_embeddings)

        del model
        image_embeddings = np.concatenate(embeds)
        print(f'Our image embeddings shape is {image_embeddings.shape}')
        del embeds
        gc.collect()
        return image_embeddings

    def getPrediction(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getImageEmbeddings()
            torch.save(image_embeddings, './image_embeddings.pt')
        else: # 加载之前计算好保存的
            image_embeddings = torch.load('./image_embeddings.pt')
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=4, KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions