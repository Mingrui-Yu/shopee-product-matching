from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F

from transformers import (BertTokenizer, AutoConfig, AutoModel)

import utils

# reference: https://www.kaggle.com/slawekbiel/resnet18-0-772-public-lb


# --------------------------------------------------------------------------------------------
BERT_PATH = './models/bertindo15g'
bert_model_file = './models/bert_indo_val0.pth'

class Params:
    need_calc_embeddings = True
    batch_size = 256
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------------------------------------------------------------
class BertTextModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, x):
        output = self.bert_model(*x)
        return output.last_hidden_state[:,0,:]


# --------------------------------------------------------------------------------------------
def load_bert_model(fname):
    model = AutoModel.from_config(AutoConfig.from_pretrained(BERT_PATH))
    state = torch.load(fname)
    model.load_state_dict(state)
    return BertTextModel(model).cuda().eval()


# --------------------------------------------------------------------------------------------
def string_escape(s, encoding='utf-8'):
    return s.encode('latin1').decode('unicode-escape').encode('latin1').decode(encoding)



# # --------------------------------------------------------------------------------------------
# class TitleTransform(Transform):
#     def __init__(self):
#         super().__init__()
#         self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
               
#     def encodes(self, row):
#         text = row.title
#         text = string_escape(text)
#         encodings = self.tokenizer(text, padding = 'max_length', max_length=100, truncation=True,return_tensors='pt')
#         keys =['input_ids', 'attention_mask', 'token_type_ids'] 
#         return tuple(encodings[key].squeeze() for key in keys)


# # --------------------------------------------------------------------------------------------
# def get_text_dls(df):
#     tfm = TitleTransform()

#     data_block = DataBlock(
#         blocks = (TransformBlock(type_tfms=tfm), 
#                   CategoryBlock(vocab=df.label_group.to_list())),
#         # splitter=ColSplitter(),
#         # splitter=None,
#         get_y=ColReader('label_group'),
#         )
#     return  (data_block.dataloaders(df, bs=1))


# ----------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        text = self.df.title[index]
        text = string_escape(text)
        encodings = self.tokenizer(text, padding = 'max_length', max_length=100, truncation=True,return_tensors='pt')
        keys =['input_ids', 'attention_mask', 'token_type_ids'] 
        return tuple(encodings[key].squeeze() for key in keys), self.df.label_group[index]



# --------------------------------------------------------------------------------------------
def embs_from_model(model, dl):
    all_embs = []
    all_ys=[]
    i = 0
    for batch in tqdm(dl):
        bx, by = batch
        for i in range(len(bx)):
            bx[i] = bx[i].cuda()
        by = by.cuda()

        with torch.no_grad():
            embs = model(bx)
            all_embs.append(embs.half())
        all_ys.append(by)

    all_embs = F.normalize(torch.cat(all_embs))
    return all_embs, torch.cat(all_ys)



# --------------------------------------------------------------------------------------------
class MatchingBert(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def test(self):
        text_dataset = TextDataset(self.dataset.df)
        print(text_dataset[0])

    def getTextEmbeddings(self):
        text_loader = torch.utils.data.DataLoader(
            TextDataset(self.dataset.df),
            batch_size=Params.batch_size,
            num_workers=Params.num_workers
        )
        bert_embs, ys = embs_from_model(load_bert_model(bert_model_file), text_loader)
        # bert_embs, ys = embs_from_model(load_bert_model(bert_model_file), get_text_dls(self.dataset.df).valid)
        return bert_embs.detach().cpu().numpy()

    def getPrediction(self):
        need_calc_embeddings = Params.need_calc_embeddings
        if need_calc_embeddings: # 计算并保存
            image_embeddings = self.getTextEmbeddings()
            np.save('./bert_text_embeddings.npy', image_embeddings)
        else: # 加载之前计算好保存的
            image_embeddings = np.load('./bert_text_embeddings.npy')
        
        image_predictions = utils.get_image_neighbors(self.dataset.df, image_embeddings, threshold=0.5, \
                                KNN=50 if len(self.dataset.df)>3 else 3)

        return image_predictions



if __name__ == '__main__':
    # Bert
    # matcher = MatchingBert(shopee_data)
    # # matcher.test()
    # shopee_data.df['image_predictions'] =  matcher.getPrediction()
    # shopee_data.df['image_precision'],  shopee_data.df['image_recall'],  shopee_data.df['image_f1'] \
    #         = utils.score( shopee_data.df['matches'],  shopee_data.df['image_predictions'])
    # image_precision =  shopee_data.df['image_precision'].mean()
    # image_recall =  shopee_data.df['image_recall'].mean()
    # image_f1 =  shopee_data.df['image_f1'].mean()
    # print(image_precision,image_recall,image_f1)


