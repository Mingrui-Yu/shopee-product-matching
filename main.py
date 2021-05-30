import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import ToGray

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

import timm
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
from matching_Bert import MatchingBert
from matching_SIFT import MatchingSIFT
from matching_SIFT_opencv import MatchingSIFTopencv
import tf_idf
import count
import utils

# ----------------------------------------------------------------------
class GlobalParams:
    img_size = 64 # reize the original image to img_size * img_size
    n_channel = 1 # 3 if color, 1 if gray





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

        if GlobalParams.n_channel == 1:
            image = image[0, :, :]

        return image, torch.tensor(1)


# ----------------------------------------------------------------------
class ShopeeDataset(object):
    def __init__(self):
        pass

    def readDataCsv(self):
        self.df = pd.read_csv('./shopee-product-matching/train.csv')
        self.image_paths = './shopee-product-matching/train_images/' + self.df['image']
        

    def addMatchesGroundTruth(self):
        tmp = self.df.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df['matches'] = self.df['label_group'].map(tmp)
        self.df['matches'] = self.df['matches'].apply(lambda x: ' '.join(x))
        print(self.df.head())

    def loadImageDataset(self,):
        '''
        全数据集。图片在image_dataset
        '''
        transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToGray(p=1.0),
                ToTensorV2(p=1.0)])
        self.image_dataset = ImageDataset(self.image_paths, transforms)


    def loadImageDataset_train_test(self):# 这个函数之后需要整合一下 因为分类之后不存在image_dataset 会有training_image_dataset和testing_image_dataset
        '''
        训练和测试数据集。图片在training_image_dataset和testing_image_dataset
        '''        
        if GlobalParams.n_channel == 3:
            transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)])
        else:
            transforms = albumentations.Compose([
                albumentations.Resize(GlobalParams.img_size, GlobalParams.img_size, always_apply=True),
                albumentations.Normalize(),
                ToGray(p=1.0),
                ToTensorV2(p=1.0)])

        self.image_dataset = ImageDataset(self.image_paths, transforms)
        self.training_image_dataset = ImageDataset(self.training_image_paths, transforms)
        self.testing_image_dataset = ImageDataset(self.testing_image_paths, transforms)


    def addSplits(self, test_group=0):
        '''
        总共分成5组,test_group指定第几组是验证集
        训练集在self.training_image_dataset
        验证集在self.testing_image_dataset
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
        self.df['is_test'] = self.df['split'] == test_group

        # print(self.df)

        # training的matches需要全部是训练集里面的图
        self.df_training = self.df[self.df['is_test']==False].drop('matches',axis=1)
        tmp = self.df_training.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df_training['matches'] = self.df_training['label_group'].map(tmp)
        self.df_training['matches'] = self.df_training['matches'].apply(lambda x: ' '.join(x))
        self.df_training = self.df_training.reset_index()
        # print(self.df_training.head(20))

        # testing的matches也需要全部是训练集里面的图
        self.df_testing = self.df[self.df['is_test']==True].drop('matches',axis=1)
        self.df_testing['matches'] = self.df_testing['label_group'].apply(lambda x: tmp.setdefault(x,[]))
        self.df_testing['matches'] = self.df_testing['matches'].apply(lambda x: ' '.join(x))
        self.df_testing = self.df_testing.reset_index()
        # print(self.df_testing.head(20))

        self.training_image_paths = './shopee-product-matching/train_images/' + self.df_training['image']
        self.testing_image_paths = './shopee-product-matching/train_images/' + self.df_testing['image']

        print('The training and testing datasets are now available!!!\n %d images in train and %d images in test'%(len(self.training_image_paths),len(self.testing_image_paths)))


    def addSplits_no2inTest(self, test_group=0):
        '''
        验证集中没有2个的，2个的全在训练集
        总共分成5组,test_group指定第几组是验证集
        训练集在self.training_image_dataset
        验证集在self.testing_image_dataset
        训练集df在self.df_training
        测试集df在self.df_testing
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
        self.df['is_test'] = self.df['split'] == test_group

        # print(self.df.head(15))

        # training的matches需要全部是训练集里面的图
        self.df_training = self.df[self.df['is_test']==False].drop('matches',axis=1)
        tmp = self.df_training.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df_training['matches'] = self.df_training['label_group'].map(tmp)
        self.df_training['matches'] = self.df_training['matches'].apply(lambda x: ' '.join(x))
        self.df_training = self.df_training.reset_index()
        # print(self.df_training.head(15))

        # testing的matches需要全部是测试集里面的图
        self.df_testing = self.df[self.df['is_test']==True].drop('matches',axis=1)
        tmp = self.df_testing.groupby(['label_group'])['posting_id'].unique().to_dict()
        self.df_testing['matches'] = self.df_testing['label_group'].apply(lambda x: tmp.setdefault(x,[]))
        self.df_testing['matches'] = self.df_testing['matches'].apply(lambda x: ' '.join(x))
        self.df_testing = self.df_testing.reset_index()

        # self.df.to_csv("./temp.csv", index=False)

        # print("!!!!!!!!!!!!!!!", self.image_paths[11])
        self.training_image_paths = './shopee-product-matching/train_images/' + self.df_training['image']
        # print("!!!!!!!!!!!!!!!", self.training_image_paths[11])
        self.testing_image_paths = './shopee-product-matching/train_images/' + self.df_testing['image']
        
        





# ----------------------------------------------------------------------
if __name__ == '__main__':



    shopee_data = ShopeeDataset()
    shopee_data.readDataCsv()
    shopee_data.addMatchesGroundTruth()
    shopee_data.loadImageDataset()
    shopee_data.addSplits_no2inTest(test_group=1)
    shopee_data.loadImageDataset_train_test()
    # print('The training and testing datasets are now available!!!\n %d images in train and %d images in test'%(len(self.training_image_paths),len(self.testing_image_paths)))

    print("finish data preparation.")

    # efficientnet 
    # matcher = MatchingNN(shopee_data)
    # shopee_data.df['image_predictions'] = matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)


    # # efficientnet 在训练集测试集上的版本
    # GlobalParams.img_size = 512 #注意要在全局里面改，因为前面数据处理已经用到了global然后别忘了把matching_NN里面的Params.batch_size改小一点，否则内存会超出
    # matcher = MatchingNN(shopee_data)
    # shopee_data.df_testing['image_predictions'] = matcher.getPrediction_testDataset()
    # shopee_data.df_testing['image_precision'],  shopee_data.df_testing['image_recall'],  shopee_data.df_testing['image_f1'] \
    #         = utils.score( shopee_data.df_testing['matches'],  shopee_data.df_testing['image_predictions'])
    # image_precision =  shopee_data.df_testing['image_precision'].mean()
    # image_recall =  shopee_data.df_testing['image_recall'].mean()
    # image_f1 =  shopee_data.df_testing['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)

    # # text using tf_idf
    # shopee_data.df['text_predictions'] = tf_idf.getTextPredictions(shopee_data.df, max_features=25000)
    # shopee_data.df['text_precision'],shopee_data. df['text_recall'], shopee_data.df['text_f1'] \
    #         = utils.score(shopee_data.df['matches'], shopee_data.df['text_predictions'])
    # text_precision = shopee_data.df['text_precision'].mean()
    # text_recall = shopee_data.df['text_recall'].mean()
    # text_f1 = shopee_data.df['text_f1'].mean()
    # print(text_precision, text_recall, text_f1)

    # # text using tf_idf 在测试集上的版本
    # shopee_data.df_testing['text_predictions'] = tf_idf.getTextPredictions(shopee_data.df_testing, max_features=25000)
    # shopee_data.df_testing['text_precision'],shopee_data. df_testing['text_recall'], shopee_data.df_testing['text_f1'] \
    #         = utils.score(shopee_data.df_testing['matches'], shopee_data.df_testing['text_predictions'])
    # text_precision = shopee_data.df_testing['text_precision'].mean()
    # text_recall = shopee_data.df_testing['text_recall'].mean()
    # text_f1 = shopee_data.df_testing['text_f1'].mean()
    # print(text_precision, text_recall, text_f1)
    # # 0.9729123396058854 0.5233578231372081 0.6274635450792525  yxj05300910 test_label=0
    # # 0.9728739318863272 0.5142002427732281 0.6171707403072557 yxj05301334 test_label=1


    # # text using count
    # shopee_data.df['text_predictions'] = count.getTextPredictions(shopee_data.df, max_features=25000)
    # shopee_data.df['text_precision'],shopee_data. df['text_recall'], shopee_data.df['text_f1'] \
    #         = utils.score(shopee_data.df['matches'], shopee_data.df['text_predictions'])
    # text_precision = shopee_data.df['text_precision'].mean()
    # text_recall = shopee_data.df['text_recall'].mean()
    # text_f1 = shopee_data.df['text_f1'].mean()
    # print(text_precision, text_recall, text_f1)

    # # text using count 在测试集上的版本
    # shopee_data.df_testing['text_predictions'] = count.getTextPredictions(shopee_data.df_testing, max_features=25000)
    # shopee_data.df_testing['text_precision'],shopee_data. df_testing['text_recall'], shopee_data.df_testing['text_f1'] \
    #         = utils.score(shopee_data.df_testing['matches'], shopee_data.df_testing['text_predictions'])
    # text_precision = shopee_data.df_testing['text_precision'].mean()
    # text_recall = shopee_data.df_testing['text_recall'].mean()
    # text_f1 = shopee_data.df_testing['text_f1'].mean()
    # print(text_precision, text_recall, text_f1)
    # # 0.965315591026893 0.5569886931925548 0.6544882110954147 yxj05300912


    # # combine image and text
    # shopee_data.df['joint_predictions'] = shopee_data.df.apply(utils.combine_predictions, axis=1)
    # shopee_data.df['joint_precision'], shopee_data.df['joint_recall'], shopee_data.df['joint_f1'] \
    #         = utils.score(shopee_data.df['matches'], shopee_data.df['joint_predictions'])
    # joint_precision = shopee_data.df['joint_precision'].mean()
    # joint_recall = shopee_data.df['joint_recall'].mean()
    # joint_f1 = shopee_data.df['joint_f1'].mean()
    # print(joint_precision,joint_recall,joint_f1)

    # # PCA
    # image_shape = (GlobalParams.n_channel, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingPCA(shopee_data, image_shape)
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)

    # # Bert 
    # matcher = MatchingBert(shopee_data.df)
    # # matcher.test()
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)
    # # 0.9553560075421472 0.6890239138269245 0.7541074948730916 yxj05301005

    # # Bert 在测试集上的版本
    # precision_log = []
    # recall_log = []
    # f1_log = []
    # for k in range(5):
    #     shopee_data.addSplits_no2inTest(k)
    #     shopee_data.loadImageDataset_train_test()
    #     matcher = MatchingBert(shopee_data.df_testing)
    #     # matcher.test()
    #     shopee_data.df_testing['image_predictions'] =  matcher.getPrediction()
    #     shopee_data.df_testing['image_precision'],  shopee_data.df_testing['image_recall'],  shopee_data.df_testing['image_f1'] \
    #             = utils.score( shopee_data.df_testing['matches'],  shopee_data.df_testing['image_predictions'])
    #     image_precision =  shopee_data.df_testing['image_precision'].mean()
    #     image_recall =  shopee_data.df_testing['image_recall'].mean()
    #     image_f1 =  shopee_data.df_testing['image_f1'].mean()
    #     print(image_precision,image_recall,image_f1)
    #     precision_log.append(image_precision)
    #     recall_log.append(image_recall)
    #     f1_log.append(image_f1)
    # print('The mean precision,recall,f1 score are:')
    # print(np.mean(precision_log),np.mean(recall_log),np.mean(f1_log))
    # # 0.9863738517571777 0.47886531884312256 0.5893483368415985 yxj05301015 addSplits_no2inTest
    # # 0.9942057928791463 0.6619027988917938 0.7415711406499854 yxj05301040 5次取平均

    # # PCA 在训练集和测试集上跑的版本
    # image_shape = (GlobalParams.n_channel, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingPCA(shopee_data, image_shape)
    # shopee_data.df_testing['image_predictions'] =  matcher.getPrediction_testDataset()
    # shopee_data.df_testing['image_precision'],  shopee_data.df_testing['image_recall'],  shopee_data.df_testing['image_f1'] \
    #         = utils.score( shopee_data.df_testing['matches'],  shopee_data.df_testing['image_predictions'])
    # image_precision =  shopee_data.df_testing['image_precision'].mean()
    # image_recall =  shopee_data.df_testing['image_recall'].mean()
    # image_f1 =  shopee_data.df_testing['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)


    # # SIFT using kornia
    # image_shape = (GlobalParams.n_channel, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingSIFT(shopee_data, image_shape)
    # # matcher = MatchingSIFT(shopee_data.image_dataset, shopee_data.df , image_shape)
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)

    # # # SIFT using kornia 在测试集上的版本
    # image_shape = (GlobalParams.n_channel, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingSIFT(shopee_data, image_shape)
    # # matcher = MatchingSIFT(shopee_data.image_dataset, shopee_data.df , image_shape)
    # shopee_data.df_testing['image_predictions'] =  matcher.getPrediction_test()
    # shopee_data.df_testing['image_precision'],  shopee_data.df_testing['image_recall'],  shopee_data.df_testing['image_f1'] \
    #         = utils.score( shopee_data.df_testing['matches'],  shopee_data.df_testing['image_predictions'])
    # image_precision =  shopee_data.df_testing['image_precision'].mean()
    # image_recall =  shopee_data.df_testing['image_recall'].mean()
    # image_f1 =  shopee_data.df_testing['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)

    # # SIFT using opencv
    # matcher = MatchingSIFTopencv(shopee_data)
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)
