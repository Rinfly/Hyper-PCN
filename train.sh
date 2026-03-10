# Train on PCN
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -m torch.distributed.launch --master_port=13231 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/PCN_models/Hyper_PCN.yaml --exp_name pcn --val_freq 10 --val_interval 50 

# Train on ShapeNet55
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -m torch.distributed.launch --master_port=13231 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet55_models/Hyper_PCN.yaml --exp_name shapenet55 --val_freq 10 --val_interval 50

# Train on ShapeNet34
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -m torch.distributed.launch --master_port=13231 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet34_models/Hyper_PCN.yaml --exp_name shapenet34 --val_freq 10 --val_interval 50

# Train on MVP
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -m torch.distributed.launch --master_port=13231 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/MVP_models/Hyper_PCN.yaml --exp_name mvp --val_freq 10 --val_interval 50