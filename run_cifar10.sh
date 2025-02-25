python3 -m torch.distributed.run --nproc_per_node=1 train.py \
  --data-path /shared_data/cifar10/ \
  --output_dir /storage/twist/logs/ \
  --aug barlow \
  --batch-size 256 \
  --dim 128 \
  --hid_dim 128 \
  --epochs 800 \
  --img_size 32 \
  --img_size_small 18 \
  --dataset cifar10 \
  --backbone resnet18 \
  --min1 0.75 --max1 1.0 \
  --min2 0.3 --max2 0.74 \
  --lr 0.5 \
  --lam1 0.0 \
  --lam2 1.0 \
  --tau 0.1 \
  #--loss_type PAWSLoss
