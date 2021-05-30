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



## 文件说明

所有代码架构面向对象，要注意跑之前GlobalParam的设置和对应类文件中batch_size、生成embedding等设置！

main.py主文件，将其中对应部分取消注释即可运行程序

matching_PCA.py为PCA类、函数

matching_SIFT.py为简化SIFT类、函数

matching_SIFT_opencv.py为SIFT类、函数（要跑很久）

matching_BERT.py为BERT类、函数

matching_NN.py为EfficientNet类、函数

count.py为词频类、函数

tf_idf.py为tf_idf类、函数

utils为常用函数

train_on_trainset.py为在训练数据集上训练的文件



