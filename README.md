# ResNet34 Implementation

## Purpose

This project is an implementation of the ResNet34 model from the paper "Deep Residual Learning for Image Recognition." This particular implementation started off with defining many base operations such as Conv2D and BatchNorm2D only using operations such as `einsum()` and `as_strided()`.  

Based on this configuration, the model achieves a 72% accuracy on the ImageNet dataset.

## Setup

If you wish to run this project, ensure (Miniconda)[] is installed on your machine and if you are on macOS or Linux you can run the following:

```bash
ENV_PATH=./resnet34/.env/
conda create -p $ENV_PATH python=3.9 -y
conda install -p $ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

If you are on Windows, you can run this:

```bash
$env:ENV_PATH='c:\users\<user_name>\resnet34\.env'
conda create -p $env:ENV_PATH python=3.9 -y
conda install -p $env:ENV_PATH pytorch=1.12.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

## Acknowledgements

Much of this implementation was guided by a program created by Redwood Research. Many thanks to Redwood for creating this program and serving as a stepping stone for this implenentation.
