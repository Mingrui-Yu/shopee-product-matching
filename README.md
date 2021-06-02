# shopee-product-matching

## Installation
Create a virtual env in anaconda: [Rapids HomePage](https://rapids.ai/start.html#rapids-release-selector)

For cuda 10.2:
```
# need to use the default conda source

conda create -n rapids-0.19 -c rapidsai -c nvidia -c conda-forge \
    rapids-blazing=0.19 python=3.7 cudatoolkit=10.2
```
Install PyTorch in the virtual env: [PyTorch](https://pytorch.org/)

Install requirements in the virtual env:
```
pip install -r requirements.txt
```
Clone the repo:
```
git clone https://github.com/Mingrui-Yu/shopee-product-matching.git
```

Download the Bert pretrained model to the folder:

```
cd shopee-product-matching
wget https://cloud.tsinghua.edu.cn/f/1914a30db3c942a2b2bd/?dl=1 -O Bert_models.zip
```

and then unzip the file.



Download the efficientnet model **trained by HuangRui** to the folder:
```
cd shopee-product-matching
wget https://cloud.tsinghua.edu.cn/f/b378be4ef2844d389ad4/?dl=1 -O acrface_models.zip
```


and then unzip the file.





## What are contained in this file:

shopee-product-matching
├── count.py: functions to get Text Predictions using words frequency
├── main.py: classes to prepare the data and main loop
├── matching_Bert.py: functions to get Text Predictions using BERT
├── matching_NN.py: classes and functions to predict the image using EfficientNet
├── matching_PCA.py: classes and functions to predict the image using PCA
├── matching_SIFT.py: classes and functions to predict the image using simplified SIFT
├── matching_SIFT_opencv.py: classes and functions to predict the image using SIFT (opencv)
├── ref_code.py: the code snippet we referred to. https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728#About-Notebook
├── ref_code2.py:the code snippet we refered to. https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728#About-Notebook

├── requirements.txt: environment requirements

├── tf_idf.py: classes and functions to get Text Predictions using TFIDF
├── train.py: code for training the EfficientNet on all dataset.

├── train_on_trainset.py: code for training the EfficientNet on training dataset.
├── utils.py: functions for get precision, recall, F1 score

**Notes**: 

* before running the code, put the [data folder](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/) in this folder

* if you want to train the EfficientNet, just run the `train.py` or `train_on_trainset.py`, remember to change the batch_size to adapt to your computation ability(by default the batch size is 24). Remember to create a folder named model1 in this folder.
* if you want to get the prediction results using any one of the methods, uncomment the according code snippets in `main.py`  and run. Remember to change `GlobalParam.img_size` in `main.py` and `Params.batch_size` in according function file before running.





