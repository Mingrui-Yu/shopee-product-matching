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
import pdb


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

def read_dataset():

    df = pd.read_csv('./shopee-product-matching/train.csv')
    df_cu = cudf.DataFrame(df)
    image_paths = './shopee-product-matching/train_images/' + df['image']

    return df, df_cu, image_paths

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG.seed)

def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def precision(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    precision = intersection/len_y_pred
    return precision

def recall(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    recall = intersection/len_y_true
    return recall

def score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    precision = intersection/len_y_pred
    recall = intersection/len_y_true
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return precision,recall,f1

def combine_predictions(row):
    image_prediction = row['image_predictions'].split()
    text_prediction = row['text_predictions'].split()
#     pdb.set_trace()
    x = np.concatenate([np.array(image_prediction), np.array(text_prediction)])  
    return ' '.join( np.unique(x) )


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
#         print(x.shape)
        x = self.pooling(x).view(batch_size, -1)  #(2, 1792)
#         print(x.shape)
#         pdb.set_trace()

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x

def get_image_neighbors(df, embeddings, KNN=50):

    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    threshold = 4
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
#         posting_ids = df['posting_id'].iloc[ids].values
        posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)
    
#     pdb.set_trace()
    del model, distances, indices
    gc.collect()
    return df, predictions

def get_test_transforms():
    return albumentations.Compose([
        albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
    ])

class ShopeeDataset(Dataset):

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

def get_image_embeddings(image_paths):

    model = ShopeeModel(pretrained=True).to(CFG.device)
#     model = ShopeeModel(pretrained=False).to(CFG.device)
#     model.load_state_dict(torch.load(CFG.model_path))
    model.eval()

    image_dataset = ShopeeDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            features = model(img,label)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings

def get_text_predictions(df, max_features=25_000):
    
    model = TfidfVectorizer(stop_words='english',
                            binary=True,
                            max_features=max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()

    print('Finding similar titles...')
    CHUNK = 1024 * 4
    CTS = len(df) // CHUNK
    if (len(df)%CHUNK) != 0:
        CTS += 1

    preds = []
    for j in range( CTS ):
        a = j * CHUNK
        b = (j+1) * CHUNK
        b = min(b, len(df))
        print('chunk', a, 'to', b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T
#         pdb.set_trace()
        for k in range(b-a):
            IDX = cupy.where(cts[k,]>0.6)[0] #0.75
#             o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
            o = ' '.join(df.iloc[cupy.asnumpy(IDX)].posting_id.values)
            preds.append(o)

    del model,text_embeddings
    gc.collect()
    return preds

df,df_cu,image_paths = read_dataset()
tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
df['matches'] = df['label_group'].map(tmp)
df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
df.head()

text_predictions = get_text_predictions(df, max_features=25_000)
df['text_predictions'] = text_predictions

# df['text_precision'] = precision(df['matches'], df['text_predictions'])
# df['text_recall'] = recall(df['matches'], df['text_predictions'])
# df['text_f1'] = f1_score(df['matches'], df['text_predictions'])

df['text_precision'],df['text_recall'],df['text_f1'] = score(df['matches'], df['text_predictions'])
text_precision = df['text_precision'].mean()
text_recall = df['text_recall'].mean()
text_f1 = df['text_f1'].mean()
print(text_precision,text_recall,text_f1)

# save好了注释掉
image_embeddings = get_image_embeddings(image_paths.values)
torch.save(image_embeddings, './image_embeddings.pt')

image_embeddings = torch.load('./image_embeddings.pt')
df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=50 if len(df)>3 else 3)
df['image_predictions'] = image_predictions

# df['image_f1'] = score(df['matches'], df['image_predictions'])
# image_f1 = df['image_f1'].mean()
# print(image_f1)

df['image_precision'],df['image_recall'],df['image_f1'] = score(df['matches'], df['image_predictions'])
image_precision = df['image_precision'].mean()
image_recall = df['image_recall'].mean()
image_f1 = df['image_f1'].mean()
print(image_precision,image_recall,image_f1)

df['joint_predictions'] = df.apply(combine_predictions, axis=1)
# df['joint_f1'] = f1_score(df['matches'], df['joint_predictions'])
# joint_f1 = df['joint_f1'].mean()
# print(joint_f1)
df['joint_precision'],df['joint_recall'],df['joint_f1'] = score(df['matches'], df['joint_predictions'])
joint_precision = df['joint_precision'].mean()
joint_recall = df['joint_recall'].mean()
joint_f1 = df['joint_f1'].mean()
print(joint_precision,joint_recall,joint_f1)