## Installation

We use Ubuntu 22.04/24.04, Python 3.9, PyTorch 2.1.1 and CUDA 11.8 for this project. The extensions Chamfer and PointNet2 are compiled with GCC 9. The model is trained on NVIDIA RTX 3090 GPUs.

You may refer to the instructions below to set up the environment and install the dependencies.

```shell
git clone https://github.com/Rinfly/Hyper-PCN.git
cd Hyper-PCN
conda create -n hyper-pcn python=3.9
conda activate hyper-pcn
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
sh extensions/install.sh
```

## Acknowledgement

This code is built upon [PoinTr](https://github.com/yuxumin/PoinTr). We are also grateful for the open-source code of [DeepHypergraph](https://github.com/iMoonLab/DeepHypergraph), [GRNet](https://github.com/hzxie/GRNet), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [DGCNN](https://github.com/WangYueFt/dgcnn) and [SymmCompletion](https://github.com/HKUST-SAIL/SymmCompletion). 