# test on PCN
CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts /path/to/pcn_ckpt.pth --config ./cfgs/PCN_models/Hyper_PCN.yaml --test_interval 50 --exp_name test_ckpt

# test on ShapeNet55
# CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts /path/to/shapenet55_ckpt.pth --config ./cfgs/ShapeNet55_models/Hyper_PCN.yaml --test_interval 50 --exp_name test_ckpt

# test on ShapeNet34
# CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts /path/to/shapenet34_ckpt.pth --config ./cfgs/ShapeNet34_models/Hyper_PCN.yaml --test_interval 50 --exp_name test_ckpt

# test on ShapeNet Unseen21
# CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts /path/to/shapenet34_ckpt.pth --config ./cfgs/ShapeNetUnseen21_models/Hyper_PCN.yaml --test_interval 50 --exp_name test_ckpt

# test on MVP
# CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts /path/to/mvp_ckpt.pth --config ./cfgs/MVP_models/Hyper_PCN.yaml --test_interval 50 --exp_name test_ckpt