cd ../
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
            --nproc_per_node=1 --master_port 1622 --use_env main.py  \
            --model deit_ms_base_patch16_224  \
            --batch-size 64 \
            --data-path /home/datasets/imagenet  \
            --data-set IMNET \
            --input-size 224  \
            --exp_name test_speed \
            --num_workers 10 \
            --ms_policy config/policy_tiny-8bist.txt \
            --train_throughput

