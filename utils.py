from cuml.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc


# ------------------------------------------
def get_image_neighbors(df, embeddings, threshold=4, KNN=50):

    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
#         posting_ids = df['posting_id'].iloc[ids].values
        posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)
    
#     pdb.set_trace()
    del model, distances, indices
    gc.collect()  # ???
    return predictions

# -------------------------------------------
def score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    precision = intersection/len_y_pred
    recall = intersection/len_y_true
    f1 = 2 * intersection / (len_y_pred + len_y_true)

    print(y_true[0])
    print(y_pred[0])



    return precision,recall,f1


# -------------------------------------------
def combine_predictions(row):
    image_prediction = row['image_predictions'].split()
    text_prediction = row['text_predictions'].split()
    # pdb.set_trace()
    x = np.concatenate([np.array(image_prediction), np.array(text_prediction)])  
    return ' '.join( np.unique(x) )