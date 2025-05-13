#! /bin/bash


WORK_DIR=${PROJECT_PATH}

DATASET_LIST=("ml-1m")

EMBEDDING_MODEL_PATH=${MODEL_DIR}/gte-Qwen2-1.5B-instruct
STRONG_MODEL_PATH=${MODEL_DIR}/Qwen2.5-14B-Instruct-AWQ
WEAK_MODEL_PATH=${MODEL_DIR}/Llama-3.2-3B-Instruct
PREDICTOR_MODEL_PATH=${MODEL_DIR}/Llama-3.2-3B-Instruct


# 1. Generate User Simulation Scene
for dataset in $DATASET_LIST; do
    python usermirrorer/scene_sampling.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --max_exposure_length 5 \
        --min_exposure_length 5 \
        --sample_nums 256 \
        --eval_set \
        --embedding_model_path $EMBEDDING_MODEL_PATH \
        --length_filtering 6144 \
        --tokenizer_path $WEAK_MODEL_PATH

    python usermirrorer/scene_sampling.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --max_exposure_length 12 \
        --min_exposure_length 2 \
        --sample_nums 1024 \
        --embedding_model_path $EMBEDDING_MODEL_PATH

    # 2. Generate Decision-making Process

    python usermirrorer/gen_decisions.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --model_path $STRONG_MODEL_PATH \
        --version strong \
        --gpu_ids "0,1"
    wait


    python usermirrorer/gen_decisions.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --model_path $WEAK_MODEL_PATH \
        --version weak \
        --gpu_ids "0,1"
    wait

    # 3. Uncertainty Estimation

    python usermirrorer/behavior_pred.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --model_path $PREDICTOR_MODEL_PATH \
        --version strong \
        --gpu_device "0"

    python usermirrorer/behavior_pred.py \
        --dataset $dataset \
        --project_path $WORK_DIR \
        --model_path $PREDICTOR_MODEL_PATH \
        --version weak \
        --gpu_device "0"
done

# 4. Data Filtering
python usermirrorer/data_filtering.py \
        --config_name UserMirrorer \
        --project_path $WORK_DIR \
        --datasets $DATASET_LIST
