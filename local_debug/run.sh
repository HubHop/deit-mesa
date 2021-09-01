cd ../
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
            --nproc_per_node=1 --master_port 1622 --use_env main.py  \
            --model deit_ms_tiny_patch16_224  \
            --batch-size 256 \
            --data-path /home/datasets/cifar100  \
            --data-set CIFAR \
            --input-size 224  \
            --exp_name channel_wise_ema_0.9 \
            --num_workers 10 \
            --ms_policy config/policy_tiny-8bit.txt 