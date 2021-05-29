import cv2

from tqdm import tqdm

import numpy as np
import pandas as pd

import utils




# ----------------------------------------------------------------------
class Params:
    batch_size = 2048
    num_workers = 4
    need_calc_embeddings = True
    n_keypoints = 50



class MatchingSIFTopencv(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_keypoints = Params.n_keypoints


    def getImageEmbeddings(self):
        num_img = len(self.dataset.image_paths)
        # num_img = 10

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.n_keypoints)

        image_embeddings = np.zeros((num_img, self.n_keypoints * 128))
        for i in tqdm(range(num_img)):
            image = cv2.imread(self.dataset.image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints, descriptors = sift.detectAndCompute(image, None)
            try:
                if descriptors.shape[0] > self.n_keypoints:
                    descriptors = descriptors[ : self.n_keypoints, :]
                image_embeddings[i][ : descriptors.size] = descriptors.reshape(-1, )
            except:
                pass

        return image_embeddings


    def getPrediction(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getImageEmbeddings()
            np.save('./sift_opencv_image_embeddings.npy', image_embeddings)
        else: # 加载之前计算好保存的
            image_embeddings = np.load('./sift_opencv_image_embeddings.npy')
        
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=0.17, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions
        
            



if __name__ == "__main__":

    m = 1
    
    # # SIFT using opencv
    # matcher = MatchingSIFTopencv(shopee_data)
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)



