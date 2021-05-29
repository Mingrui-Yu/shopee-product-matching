from cuml.feature_extraction.text import CountVectorizer
import cupy
import cudf
import cuml

from cuml.experimental.preprocessing import normalize
import numpy as np
import pandas as pd
import gc
import pdb

# -------------------------------------------------------------------------------------------------
def getTextPredictions(df, max_features=25_000):
    
    model = CountVectorizer(stop_words='english',
                            binary=True,
                            max_features=max_features)
    df_cu = cudf.DataFrame(df)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()

    # pdb.set_trace()

    text_embeddings = normalize(text_embeddings, axis=1, norm='l2')

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

        # pdb.set_trace()
        for k in range(b-a):
            IDX = cupy.where(cts[k,]>0.6)[0] #0.75
#             o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
            o = ' '.join(df.iloc[cupy.asnumpy(IDX)].posting_id.values)
            preds.append(o)

    del model,text_embeddings
    gc.collect()
    return preds