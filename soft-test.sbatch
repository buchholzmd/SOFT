#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=soft-test

module purge

## use --nv for interactive mode in order to
## pass CUDA drivers from GPU to container
singularity exec --nv \
	    --overlay $SINGULARITY_IMAGES/bayesian_attention/bayesian_attention.ext3:ro \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
        /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif\
        /bin/bash -c "source /ext3/env.sh; ./dist_train.sh 4 --data $IMAGENET --config config/SOFT_Tiny.yaml --eval_checkpoint ${CHECKPOINT_FILE} --eval"