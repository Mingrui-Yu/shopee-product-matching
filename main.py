import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

import gc
import matplotlib.pyplot as plt
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
import pdb

from matching_PCA import MatchingPCA
from matching_NN import MatchingNN


# ----------------------------------------------------------------------
class GlobalParams:
    img_size = 64 # reize the original image to img_size * img_size




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
class ShopeeDataset(object):
    def __init__(self):
        pass

    def readDataCsv(self):
        self.df = pd.read_csv('./shopee-product-matching/train.csv')
        self.df_cu = cudf.DataFrame(self.df)
        self.image_paths = './shopee-product-matching/train_images/' + self.df['image']

    def addMatchesGroundTruth(self):
        tmp = self.df.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df['matches'] = self.df['label_group'].map(tmp)
        self.df['matches'] = self.df['matches'].apply(lambda x: ' '.join(x))
        print(self.df.head())

    def loadImageDataset(self):# 这个函数之后需要整合一下 因为分类之后不存在image_dataset 会有training_dataset和valid_dataset
        transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)])
        self.image_dataset = ImageDataset(self.image_paths, transforms)

    def addSplits(self, valid_group=0):
        '''
        总共分成5组,valid_group指定第几组是验证集
        训练集在self.training_dataset
        验证集在self.valid_dataset
        '''
        grouped = self.df.groupby('label_group').size()
        # print(grouped)

        labels, sizes =grouped.index.to_list(), grouped.to_list()

        # print('group index to list',labels)

        skf = StratifiedKFold(5)
        splits = list(skf.split(labels, sizes))

        group_to_split =  dict()
        for idx in range(5):
            labs = np.array(labels)[splits[idx][1]]
            group_to_split.update(dict(zip(labs, [idx]*len(labs))))

        self.df['split'] = self.df.label_group.replace(group_to_split)
        self.df['is_valid'] = self.df['split'] == valid_group

        print(self.df)

        transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)])

        self.training_image_paths = './shopee-product-matching/train_images/' + self.df[self.df['is_valid']==False]['image']
        self.valid_image_paths = './shopee-product-matching/train_images/' + self.df[self.df['is_valid']==True]['image']
        self.training_dataset = ImageDataset(self.training_image_paths, transforms)
        self.valid_dataset = ImageDataset(self.valid_image_paths, transforms)
        print('The training and validation datasets are now available!!!\n %d images in tarin and %d images in valid'%(len(self.training_image_paths),len(self.valid_image_paths)))

        return self.df

    def addSplits_no2inValid(self, valid_group=0):
        '''
        验证集中没有2个的，2个的全在训练集
        总共分成5组,valid_group指定第几组是验证集
        训练集在self.training_dataset
        验证集在self.valid_dataset
        '''
        self.df['matches_num'] = self.df['matches'].apply(lambda x: x.count('train_')).values

        grouped = self.df[self.df['matches_num']>2].groupby('label_group').size()
        # print(grouped)
        labels, sizes = grouped.index.to_list(), grouped.to_list()
        # print('group index to list',labels)

        skf = StratifiedKFold(5)
        splits = list(skf.split(labels, sizes))

        group_to_split =  dict()
        for idx in range(5):
            labs = np.array(labels)[splits[idx][1]]
            group_to_split.update(dict(zip(labs, [idx]*len(labs))))
        # print(group_to_split)

        self.df['split'] = self.df['label_group'].map(group_to_split)# 这一步之后，matches_num=2的split列都会变成NaN

        self.df['is_valid'] = self.df['split'] == valid_group
        
        # self.df.to_csv("./temp.csv", index=False)

        transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)])

        self.training_image_paths = './shopee-product-matching/train_images/' + self.df[self.df['is_valid']==False]['image']
        self.valid_image_paths = './shopee-product-matching/train_images/' + self.df[self.df['is_valid']==True]['image']
        self.training_dataset = ImageDataset(self.training_image_paths,self.image_num, transforms)
        self.valid_dataset = ImageDataset(self.valid_image_paths, transforms)
        print('The training and validation datasets are now available!!!\n %d images in tarin and %d images in valid'%(len(self.training_image_paths),len(self.valid_image_paths)))
        
        return self.df





# ----------------------------------------------------------------------
if __name__ == '__main__':
    shopee_data = ShopeeDataset()
    shopee_data.readDataCsv()
    shopee_data.addMatchesGroundTruth()
    shopee_data.loadImageDataset()

    print("finish data preparation.")

    # efficientnet 
    matcher = MatchingNN(shopee_data)
    need_calc_embeddings = True
    if need_calc_embeddings: # 计算并保存
        image_embeddings = matcher.getImageEmbeddings()
        torch.save(image_embeddings, './image_embeddings.pt')
    else: # 加载之前计算好保存的
        image_embeddings = torch.load('./image_embeddings.pt')

    ## PCA
    # image_shape = (3, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingPCA(shopee_data, image_shape)
    # matcher.main()

    ## 验证数据集分割
    # shopee_data.addSplits()
    # shopee_data.addSplits_no2inValid()
