import torch
from torch.utils.data import Dataset,DataLoader
from kornia.feature import SIFTDescriptor

from tqdm import tqdm

import numpy as np
import pandas as pd

import utils




# ----------------------------------------------------------------------
class Params:
    batch_size = 2048
    num_workers = 4
    need_calc_embeddings = True



class MatchingSIFT(object):
    def __init__(self, dataset, image_shape):
        self.dataset = dataset
        self.image_shape = image_shape

        if self.image_shape[0] != 1:
            print("Error: input must be gray image with 1 channel !")
            return

    def getImageEmbeddings(self):
        image_loader = torch.utils.data.DataLoader(
            self.dataset.image_dataset,
            batch_size=Params.batch_size,
            num_workers=Params.num_workers
        )

        SIFT = SIFTDescriptor(self.image_shape[1], 8, 4)

        descs = np.empty((self.dataset.df.shape[0], 128))
        idx = 0
        for img_batch, label in tqdm(image_loader): 
            img_vector_batch = img_batch.reshape(-1, 1,  self.image_shape[1], self.image_shape[2])
            next_idx = idx + img_vector_batch.shape[0]
            descs_batch = SIFT(img_vector_batch) # 23x128
            descs[idx : next_idx, :] = descs_batch.detach().cpu().numpy()
            idx = next_idx

        return descs

    def getPrediction(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getImageEmbeddings()
            np.save('./sift_image_embeddings.npy', image_embeddings)
        else: # 加载之前计算好保存的
            image_embeddings = np.load('./sift_image_embeddings.npy')
        
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=0.17, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions
            



if __name__ == "__main__":
    # # SIFT using kornia
    # image_shape = (GlobalParams.n_channel, GlobalParams.img_size, GlobalParams.img_size)
    # matcher = MatchingSIFT(shopee_data, image_shape)
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)



