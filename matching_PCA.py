# implement the matching algorithm using PCA
import numpy as np
import pandas as pd
import cv2
from cuml import IncrementalPCA
import torch
import torchvision
import time
import utils
import copy
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------------------------------------------------------------
class Params:
    batch_size = 256
    num_workers = 0
    n_components = 200
    # need_calc_embeddings = False
    need_calc_embeddings = True


# ----------------------------------------------------------------------
class MatchingPCA(object):

    def __init__(self, dataset, image_shape):
        self.dataset = dataset # dataset is an object of class ShopeeDataset
        self.image_shape = image_shape
        self.n_components = Params.n_components
        self.batch_size = Params.batch_size
        self.image_num =  dataset.image_paths.shape[0]

    ## 以下两个函数是在分割数据集上用的
    def getImageEmbeddings_train_test(self):
        '''
        从train fit得到数据，transform到test数据上，输出test_reduced
        '''
        image_loader = torch.utils.data.DataLoader(
            self.dataset.training_image_dataset,
            batch_size=self.batch_size,
            num_workers=Params.num_workers
        )

        name = 'iPCA'
        estimator = IncrementalPCA(n_components=self.n_components, \
                                                                    batch_size=self.batch_size)

        # fit data to the estimator in batches
        print("Start fitting training data to incremental PCA estimator ... ")
        for img_batch, label in tqdm(image_loader): 
            img_vector_batch = img_batch.reshape(-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
            estimator.partial_fit(img_vector_batch.detach().numpy())
        
        print("Start transforming testing data using the estimator ...")
        image_loader = torch.utils.data.DataLoader(
            self.dataset.testing_image_dataset,
            batch_size=self.batch_size,
            num_workers=Params.num_workers
        )
        data_reduced = np.empty((self.dataset.df_testing.shape[0], self.n_components))
        idx = 0
        for img_batch, label in tqdm(image_loader):
            img_vector_batch = img_batch.reshape(-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
            next_idx = idx + img_vector_batch.shape[0]
            data_reduced_batch = estimator.transform(img_vector_batch.detach().numpy())
            data_reduced[idx : next_idx, :] = data_reduced_batch
            idx = next_idx
            
        # Show the quality of reduction    
        print('There are %d components.'% estimator.n_components_)
        print('The variance ratio of the largest %d components: %0.4f'%
            (estimator.n_components_, sum(estimator.explained_variance_ratio_)))

        return data_reduced

    def getPrediction_testDataset(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getImageEmbeddings_train_test()
            np.save('./pca_image_embeddings_train_test.npy', image_embeddings)
        else: # 加载之前计算好保存的
            image_embeddings = np.load('./pca_image_embeddings_train_test.npy')
        
        image_predictions = utils.get_image_neighbors(self.dataset.df_testing, image_embeddings, threshold=50, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions


    ## 以下两个函数是在全数据集上用的
    def getImageEmbeddings(self):
        image_loader = torch.utils.data.DataLoader(
            self.dataset.image_dataset,
            batch_size=self.batch_size,
            num_workers=Params.num_workers
        )

        name = 'iPCA'
        estimator = IncrementalPCA(n_components=self.n_components, \
                                                                    batch_size=self.batch_size)

        # fit data to the estimator in batches
        print("Start fitting data to incremental PCA estimator ... ")
        for img_batch, label in tqdm(image_loader):
            img_vector_batch = img_batch.reshape(-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
            estimator.partial_fit(img_vector_batch.detach().numpy())
        
        print("Start transforming data using the estimator ...")
        data_reduced = np.empty((self.image_num, self.n_components))
        idx = 0
        for img_batch, label in tqdm(image_loader):
            img_vector_batch = img_batch.reshape(-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2])
            next_idx = idx + img_vector_batch.shape[0]
            data_reduced_batch = estimator.transform(img_vector_batch.detach().numpy())
            data_reduced[idx : next_idx, :] = data_reduced_batch
            idx = next_idx
            
        # Show the quality of reduction    
        print('There are %d components.'% estimator.n_components_)
        print('The variance ratio of the largest %d components: %0.4f'%
            (estimator.n_components_, sum(estimator.explained_variance_ratio_)))

        return data_reduced

    def getPrediction(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getImageEmbeddings()
            np.save('./pca_image_embeddings.npy', image_embeddings)
        else: # 加载之前计算好保存的
            image_embeddings = np.load('./pca_image_embeddings.npy')
        
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=100, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions
