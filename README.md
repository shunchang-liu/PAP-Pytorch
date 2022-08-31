# PAP-pytorch

This is the official pytorch version repo for [Harnessing Perceptual Adversarial Patches for Crowd Counting].

[![v4o4YV.png](https://s1.ax1x.com/2022/08/31/v4o4YV.png)](https://imgse.com/i/v4o4YV)

## ATTACK

### Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.X
PyTorch: 1.4.0+


### Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

### Ground Truth

Please follow the [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch) to generate the ground truth.

### Models
In the paper, we totally use six crowd couting models with the repos as follows:

MCNN: https://github.com/CommissarMa/MCNN-pytorch

CSRNet: https://github.com/leeyeehoo/CSRNet-pytorch

CAN: https://github.com/weizheliu/Context-Aware-Crowd-Counting

BL: https://github.com/ZhihengCV/Bayesian-Crowd-Counting

DM-Count: https://github.com/cvlab-stonybrook/DM-Count

SASNet: https://github.com/TencentYoutuResearch/CrowdCounting-SASNet

Here we give the official pre-trained CSRNet. You can also use other crowd counting models. 

ShanghaiA  [Google Drive](https://drive.google.com/open?id=1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH)

ShanghaiB  [Google Drive](https://drive.google.com/open?id=1zKn6YlLW3Z9ocgPbP99oz7r2nC7_TBXK)

### Training Process

 Try `python patch_attack.py` to start training process.

For attacking the CSRNet, you may modify the follow ones:

`data_root = './data/attack_shanghai/' #the dataset root`

`model_path = './pre_trained/PartA_model.pth.tar' #the pre-trained model root`

`save_path = './results' # the results root, finally you can get the images with our adversarial patches`

For attacking other models, you need to modify the `patch_attack.py` to fit your target model. 



