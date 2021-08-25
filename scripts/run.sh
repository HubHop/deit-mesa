source activate quant
which python
nvidia-smi
python -m torch.distributed.launch \
        --nproc_per_node=8 --master_port 1622 --use_env main.py  \
        --model deit_ms_tiny_patch16_224  \
        --batch-size 128 \
        --data-path /projects/dl65/m3_imagenet  \
        --data-set CIFAR \
        --input-size 224  \
        --output_dir ./exp_cuda/full \
        --num_workers 10 \
        --ms_policy config/policy_tiny-8bit.txt