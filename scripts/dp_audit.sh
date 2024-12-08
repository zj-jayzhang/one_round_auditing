#!/bin/bash
#SBATCH --array=0-31
#SBATCH --job-name=dp_audit
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=YOUT_PATH/logs/%A_%a.out

SEED=2024  #0
NUM_SHADOW=32

# Shared HPs
AUGMULT_FACTOR="16"
LEARNING_RATE="4.0"
MAX_GRAD_NORM="1.0"

# eps=7
# BATCH_SIZE="4096"
# NUM_EPOCHS="220"
# NOISE_MULTIPLIER="3.0"



# eps = 100
BATCH_SIZE="1024"
NUM_EPOCHS="60"
NOISE_MULTIPLIER="0.4"

NUM_CANARIES=1000
NUM_POISON=0
POISON_TYPE="canary_duplicates_noisy"

# print the BATCH_SIZE, NUM_EPOCHS, NOISE_MULTIPLIER
echo "BATCH_SIZE: ${BATCH_SIZE}, NUM_EPOCHS: ${NUM_EPOCHS}, NOISE_MULTIPLIER: ${NOISE_MULTIPLIER}"

CANARY_TYPE_ALL=("label_noise") # auto, clean, ood
CANARY_TYPE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHADOW))
SHADOW_MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SHADOW))
CANARY_TYPE="${CANARY_TYPE_ALL[$CANARY_TYPE_IDX]}"
SUFFIX="_audit"
EXPERIMENT="${CANARY_TYPE}${SUFFIX}"

EXPERIMENT_DIR=YOUT_EXPERIMENT_PATH
DATA_DIR=YOUT_DATA_PATH
REPO_DIR=YOUT_REPO_PATH

echo "Running task ID ${SLURM_ARRAY_TASK_ID} for experiment ${EXPERIMENT}, shadow model ${SHADOW_MODEL_IDX}"
echo "Canaries: ${NUM_CANARIES} (${CANARY_TYPE}), poisons: ${NUM_POISON} (${POISON_TYPE})"

# Prepare environment
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"


python -u -m dp_audit --experiment-dir "${EXPERIMENT_DIR}" --experiment "${EXPERIMENT}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
    --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
    --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
    audit
    
echo "Task finished"