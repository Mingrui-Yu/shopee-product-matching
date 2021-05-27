# shopee-product-matching

## Installation
Create a virtual env in anaconda: [Rapids HomePage](https://rapids.ai/start.html#rapids-release-selector)

cuda 10.2:
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
