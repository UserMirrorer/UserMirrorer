# 🎉 UserMirrorer: Towards Preference-aligned LLM User Simulator

<a href="https://github.com/UserMirrorer/UserMirrorer/blob/master/LICENSE.md"><img src="https://img.shields.io/github/license/UserMirrorer/UserMirrorer"></a>
<a href="https://huggingface.co/datasets/MirrorUser/UserMirrorer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Training_Set-yellow"></a>
<a href="https://huggingface.co/datasets/MirrorUser/UserMirrorer-eval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval_Set-yellow"></a>
<a href="https://huggingface.co/MirrorUser/UserMirrorrer-Llama-DPO"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-red"></a>

****

This repository contains the `UserMirrorer` framework for the paper "Mirroring Users: Towards Building Preference-aligned User Simulator with Recommendation Feedback".

## 📚 Contents

- [Installation](#installation)
- [Step-by-Step Guide to Construct Dataset from Raw Data](#step-by-step-guide-to-construct-dataset-from-raw-data)
- [LLM Fine-tuning](#llm-fine-tuning)
- [Evaluation](#evaluation)

## 🚧 Installation

```bash
pip install -r requirements.txt
```

## 🚧 Step-by-Step Guide to Construct Dataset from Raw Data

### 📝 1. Create your working directory
```bash
mkdir -p <YOUR_WORKING_DIR>
```

### 📝 2. Raw Data PreProcessing
First, we need to pre-process the raw data to get a unified format as the input of `UserMirrorer` framework.

Consider each dataset has its own format, we provide the following scripts for the datasets we used in the paper.

For other datasets, you can follow the format of the provided notebooks to create your own pre-processing scripts.

Take `Movielens-1M` as an example, you can execute the notebook `preprocessing/DataProcessor_ML1M.ipynb`, following instructions to fill in the correct paths of raw data source and working directory, and pre-process the raw data.

After pre-processing, you will get a unified format of dataset as the input of `UserMirrorer` framework, including 3 files:
```shell
<YOUR_WORKING_DIR>/raws/
├── <DATASET_NAME>_user_feature.jsonl
├── <DATASET_NAME>_item_feature.jsonl
└── <DATASET_NAME>_interaction.jsonl
```

- `<DATASET_NAME>_user_feature.jsonl`: The user feature file.
- `<DATASET_NAME>_item_feature.jsonl`: The item feature file.
- `<DATASET_NAME>_interaction.jsonl`: The interaction file.

### 📝 3. Creation of User Simulation Scene

Next, we can will create the user simulation scene for each dataset.

The user simulation scene usually contains 3 parts:
- User Profile: The user profile is the user's information.
- Interaction History: The interaction history is the user's interaction history with the items.
- Exposure: The exposure to user is the items that has been shown to the user at the current time.

To complete the creation of user simulation scene, we need to:
- Design a feature construction function to construct the additional features for user simulation (e.g., the time lasts since the last interaction, which can be derived from the differece between the timestamps).
- Design a template to convert the raw features and constructed features of user and items into the text description, so that they can be read and understood by LLM.

For datasets that do not contains the exposure information, we also need to design a strategy to sample the exposure to user from the item pool, which includes:
- Design a sampling strategy to sample the exposure to user from the item pool.
- Design a strategy to filter the items that should not be included in the exposure.

We have provided the implementation of the above steps for different datasets, which can be found in the `src/strategy` folder. You can also create your own strategy for other datasets.

To create the user simulation scene, you can execute the following command:
```bash
python usermirrorer/scene_sampling.py \
    --dataset <DATASET_NAME> \                          # The name of the dataset
    --project_path <YOUR_WORKING_DIR> \                 # The path to your working directory
    --max_exposure_length <MAX_EXPOSURE_LENGTH> \       # The maximum length of the exposure
    --min_exposure_length <MIN_EXPOSURE_LENGTH> \       # The minimum length of the exposure
    --sample_nums <SAMPLE_NUMS> \                       # The number of samples to sample
    --embedding_model_path <EMBEDDING_MODEL_PATH> \     # The path to the embedding model
    --eval_set \ # Whether to create the evaluation set # Whether to create the evaluation set
```
Here the embedding model is used to embed the user and item features into a high-dimensional space, so that we can use the embedding model to calculate the similarity between the user and item features. We use `vLLM` as the backend, so you can use any other embedding model that [supports it](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html).

Following the above command, you will get the user simulation scene for the dataset, which is stored in the `<YOUR_WORKING_DIR>/dataset` folder:
```shell
<YOUR_WORKING_DIR>/dataset/
├── <DATASET_NAME>_train.jsonl
└── <DATASET_NAME>_eval.jsonl
```

### 📝 4. Generating Decision-making Process

In this step, we will generate multiple decision-making process for each user simulation scene, using a strong LLM and a weak LLM.

To generate the decision-making process, you can execute the following command:
```bash
python usermirrorer/behavior_pred.py \                   # The path to the behavior prediction script
    --dataset <DATASET_NAME> \              # The name of the dataset
    --project_path <YOUR_WORKING_DIR> \     # The path to your working directory
    --model_path <MODEL_PATH> \             # The path to the model
    --version <VERSION> \                   # The version of the model
    --gpu_device <GPU_DEVICE> \             # The GPU device to use (data parallelism)
```

After deriving the decision-making process, you will get the following files:
```shell
<YOUR_WORKING_DIR>/decisions/
├── <DATASET_NAME>_decisions_strong.jsonl
└── <DATASET_NAME>_decisions_weak.jsonl
```

Then, we employ a LLM to predict the behavior of the user based on the decision-making process. You can execute the following command:
```bash
python usermirrorer/behavior_pred.py \                   # The path to the behavior prediction script
    --dataset <DATASET_NAME> \              # The name of the dataset
    --project_path <YOUR_WORKING_DIR> \     # The path to your working directory
    --model_path <MODEL_PATH> \             # The path to the model
    --version <VERSION> \                   # The version of the model
    --gpu_device <GPU_DEVICE> \             # The GPU device to use (tensor parallelism)
```

After predicting the behavior, you will get the following files:
```shell
<YOUR_WORKING_DIR>/probs/
├── <DATASET_NAME>_probs_strong.jsonl
└── <DATASET_NAME>_probs_weak.jsonl
```

### 📝 5. Data Filtering

After generating the decision-making process and the corresponding behavior prediction, we can filter the data to get the final dataset for training.

To filter the data, you can execute the following command:
```bash
python usermirrorer/data_filtering.py \
    --project_path <YOUR_WORKING_DIR> \     # The path to your working directory   
    --config_name <CONFIG_NAME> \           # Name of the filtered dataset
    --datasets <DATASET_NAME>               # The domain included in the final dataset
```

After filtering the data, you will get the final dataset for training, which is stored in the `<YOUR_WORKING_DIR>/datasets` folder:
```shell
<YOUR_WORKING_DIR>/datasets/
└── <CONFIG_NAME>_pref.jsonl
```

## 📚 LLM Fine-tuning

The dataset derived in the previous stages can be used to fine-tune the LLM.

We use `torchtune` to fine-tune the LLM. You can refer to the `configs` folder for the fine-tuning configs.


You can also access our derived datasets and fine-tuned models on the Hugging Face:
- *Dataset*: [Train](https://huggingface.co/datasets/MirrorUser/UserMirrorer) and [Eval](https://huggingface.co/datasets/MirrorUser/UserMirrorer-eval)
- *Fine-tuned Models*: Fine-tuned based on [Qwen2.5-3B-Instruct](https://huggingface.co/MirrorUser/UserMirrorrer-Qwen-DPO) and [Llama-3.2-3B-Instruct](https://huggingface.co/MirrorUser/UserMirrorrer-Llama-DPO)

> 

We have provided an example script as `run.sh` to produce the datasets.

## 📚 Evaluation

To run the evaluation, you can execute the following command:
```bash
python usermirrorer/run_eval.py \
    --project_path <YOUR_WORKING_DIR> \     # The path to your working directory
    --model_path <MODEL_PATH> \             # The path to the model
    --input_file <INPUT_FILE> \             # The path to the input file
    --output_file <OUTPUT_FILE> \           # The path to the output file
    --mode <MODE> \                         # The mode of the evaluation
    --repeat_times <REPEAT_TIMES> \         # The number ofsampling times
```

