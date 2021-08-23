cd ../
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
            --nproc_per_node=1 --master_port 1622 --use_env main.py  \
            --model deit_ms_tiny_patch16_224  \
            --batch-size 128 \
            --data-path /home/datasets/imagenet  \
            --data-set IMNET \
            --input-size 224  \
            --output_dir ./exp_cuda/full-new-quant \
            --num_workers 10 \
            --ms_policy config/policy_tiny-8bit.txt 