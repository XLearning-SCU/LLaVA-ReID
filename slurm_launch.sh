#!/bin/bash

#SBATCH --job-name=llava-reid         # create a short name for your job
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks-per-node=4           # total number of tasks per node
#SBATCH --gres=gpu:4                  # number of gpus per node
#SBATCH --cpus-per-task=8            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=320GB                   # total memory per node
#SBATCH --time=24:00:00              # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu4090_EU        # partition of cluster
#SBATCH --output=./report/%j.log      # log file


echo "Nodelist:" $SLURM_JOB_NODELIST
echo "Number of nodes:" $SLURM_JOB_NUM_NODES
echo "tasks per node:" $SLURM_NTASKS_PER_NODE

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cpu_per_task=1
if [ -n "${SLURM_CPUS_PER_TASK}" ]; then
	cpu_per_task=$SLURM_CPUS_PER_TASK
fi
export OMP_NUM_THREADS=${cpu_per_task}
echo "OMP_NUM_THREADS="$OMP_NUM_THREADS

echo "Run started at: $(date)"

# Actual run of script
srun python -u main_train.py --config_file=config/Interactive-PEDES.yaml
#srun deepspeed train_llava_reid.py --config_file=config/Interactive-PEDES.yaml