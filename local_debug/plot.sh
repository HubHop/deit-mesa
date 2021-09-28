cd ../
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
            --nproc_per_node=1 --master_port 1622 --use_env main.py  \
            --model deit_ms_tiny_analyse  \
            --batch-size 128 \
            --data-path /home/datasets/imagenet  \
            --data-set IMNET \
            --input-size 224  \
            --exp_name torch-quant-ema-0.9-fp-forward \
            --num_workers 10 \
            --ms_policy config/policy_tiny-8bit.txt \
            --get_act_dist 
