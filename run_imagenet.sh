python3 -m torch.distributed.run --nproc_per_node=1 train.py \
  --data-path /shared_data/imagenet/ \
  --output_dir /storage/twist/logs/ \
  --aug barlow \
  --batch-size 256 \
  --dim 32768 \
  --epochs 800 \
  #--num_workers 1 \
  #--device cuda:4 
