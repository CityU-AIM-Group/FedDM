#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH -w node1
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

nvidia-smi

python main.py --dataset='./data/promise_WSS'\
    --workdir='results/prostate/box_ce_tmp'\
    --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"\
    --folders="[('img', png_transform, False), ('gt', gt_transform, True)]+[('box20', gt_transform, True)]"\
    --network='ResidualUNet'\
    --n_class=2\
    --cpu=False\
    --use_sgd=False\
    --group=False\
    --n_epoch=200\
    --l_rate=5e-4\
    --grp_regex='(Case\d+_\d+)_\d+'\
    --scheduler="DummyScheduler"\
    --scheduler_params="{}"\
    --batch_size=4\
    --client_num=4\
    --worker_steps=1 \
    --peer_learning=True \
    --ratio=0.6\
    --stop_epoch=50 \
    --seed=1

python main.py --dataset='./data/promise_WSS'\
    --workdir='results/prostate/box_ce_tmp'\
    --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"\
    --folders="[('img', png_transform, False), ('gt', gt_transform, True)]+[('box20', gt_transform, True)]"\
    --network='ResidualUNet'\
    --n_class=2\
    --cpu=False\
    --use_sgd=False\
    --group=False\
    --n_epoch=200\
    --l_rate=5e-4\
    --grp_regex='(Case\d+_\d+)_\d+'\
    --scheduler="DummyScheduler"\
    --scheduler_params="{}"\
    --batch_size=4\
    --client_num=4\
    --worker_steps=1 \
    --peer_learning=True \
    --ratio=0.6\
    --stop_epoch=50 \
    --seed=2

python main.py --dataset='./data/promise_WSS'\
    --workdir='results/prostate/box_ce_tmp'\
    --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"\
    --folders="[('img', png_transform, False), ('gt', gt_transform, True)]+[('box20', gt_transform, True)]"\
    --network='ResidualUNet'\
    --n_class=2\
    --cpu=False\
    --use_sgd=False\
    --group=False\
    --n_epoch=200\
    --l_rate=5e-4\
    --grp_regex='(Case\d+_\d+)_\d+'\
    --scheduler="DummyScheduler"\
    --scheduler_params="{}"\
    --batch_size=4\
    --client_num=4\
    --worker_steps=1 \
    --peer_learning=True \
    --ratio=0.6\
    --stop_epoch=50 \
    --seed=3



