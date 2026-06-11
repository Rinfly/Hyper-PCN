[![Paper](https://img.shields.io/badge/CVPR_2026-Paper-4B44CE?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Hyper-PCN_Hypergraph-Based_Point_Cloud_Completion_via_High-Order_Correlation_Modeling_CVPR_2026_paper.pdf)
[![Pretrained Models](https://img.shields.io/badge/Pretrained_Models-Download-2EA44F?style=for-the-badge&logo=icloud&logoColor=white)](https://cloud.tsinghua.edu.cn/d/3a25ee4ea1f145bdbd14/)

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

## Dataset

Download the datasets from the following links:

- [PCN](https://gateway.infinitescript.com/s/ShapeNetCompletion)
- [MVP](https://drive.google.com/drive/folders/1ylC-dYFM45KW4K9tPyljBSVyetazCEeH)
- [ShapeNet55/34](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK)
- [KITTI](https://drive.google.com/file/d/1OsahmUKG4J7hsOTbb45w6pbFOnJ0x4Vv)

After downloading the datasets, replace the placeholder paths in
`cfgs/dataset_configs` with the corresponding local paths.

For PCN, update `PCN.yaml`:

```yaml
PARTIAL_POINTS_PATH: /path/to/PCN/%s/partial/%s/%s/%02d.pcd
COMPLETE_POINTS_PATH: /path/to/PCN/%s/complete/%s/%s.pcd
```

For MVP, update `MVP.yaml`:

```yaml
PARTIAL_POINTS_PATH: /path/to/MVP/mvp_%s_input.h5
COMPLETE_POINTS_PATH: /path/to/MVP/mvp_%s_gt_%dpts.h5
```

For ShapeNet55/34, update `PC_PATH` in `ShapeNet-55.yaml`,
`ShapeNet-34.yaml`, and `ShapeNet-Unseen21.yaml`:

```yaml
PC_PATH: /path/to/ShapeNet55-34/shapenet_pc
```

For KITTI, update `KITTI.yaml`:

```yaml
CLOUD_PATH: /path/to/KITTI/cars/%s.pcd
BBOX_PATH: /path/to/KITTI/bboxes/%s.txt
```

## Training and Tesing

Plase refer to `train.sh` and `test.sh` for the training and testing commands.

## Acknowledgement

This code is built upon [PoinTr](https://github.com/yuxumin/PoinTr). We are also grateful for the open-source code of [DeepHypergraph](https://github.com/iMoonLab/DeepHypergraph), [GRNet](https://github.com/hzxie/GRNet), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [DGCNN](https://github.com/WangYueFt/dgcnn) and [SymmCompletion](https://github.com/HKUST-SAIL/SymmCompletion).

## Bibtex

```bibtex
@inproceedings{li2026hyper,
  title={Hyper-PCN: Hypergraph-Based Point Cloud Completion via High-Order Correlation Modeling},
  author={Li, Linfei and Tan, Pei and Li, Siqi and Zou, Changqing and Gao, Yue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={39121--39130},
  year={2026}
}
```