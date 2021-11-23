MODEL=$1
GPUS=$2
PORT=${PORT:-29500}

python -m torch.distributed.launch \
            --nproc_per_node=$GPUS --master_port $PORT --use_env main.py  \
            --model $MODEL  \
            --batch-size 128 \
            --data-path /home/datasets/cifar100  \
            --data-set CIFAR \
            --input-size 224  \
            --exp_name debug \
            --num_workers 10 \
            --ms_policy config/policy_tiny-8bit.txt
