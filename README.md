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

Download the pretrained model to the folder:
```
cd shopee-product-matching
wget https://cloud.tsinghua.edu.cn/f/1914a30db3c942a2b2bd/?dl=1 -O models.zip
```
and then unzip the file.