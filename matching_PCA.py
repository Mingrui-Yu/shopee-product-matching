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
    batch_size = 2048
    num_workers = 4
    n_components = 200
    need_calc_embeddings = False


# ----------------------------------------------------------------------
class MatchingPCA(object):

    def __init__(self, dataset, image_shape):
        self.dataset = dataset # dataset is an object of class DatasetPreparation
        self.image_shape = image_shape
        self.n_components = Params.n_components
        self.batch_size = Params.batch_size
        self.image_num =  dataset.image_paths.shape[0]

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
            img_vector_batch = img_batch.reshape(-1, 3 * self.image_shape[1] * self.image_shape[2])
            estimator.partial_fit(img_vector_batch.detach().numpy())
        
        print("Start transforming data using the estimator ...")
        data_reduced = np.empty((self.image_num, self.n_components))
        idx = 0
        for img_batch, label in tqdm(image_loader):
            img_vector_batch = img_batch.reshape(-1, 3 * self.image_shape[1] * self.image_shape[2])
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
        
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=50, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions
