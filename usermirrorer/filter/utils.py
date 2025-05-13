import os
import pandas as pd
import numpy as np
from collections import OrderedDict

np.random.seed(0)

def sampling(choice: dict, n_times: int = 5):
    prob = get_prob(choice, 5)
    choices = np.random.choice(range(len(prob)), size=n_times, p=prob)
    # choices = [np.argmax(prob)] * n_times
    return choices if n_times > 1 else choices[0]

def get_entropy(prob: np.array):
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def get_prob(choice: dict, choice_cnt: int):
    default_prob = {chr(i + 65): 1e-10 for i in range(choice_cnt)}
    default_prob.update(choice)
    choice = OrderedDict(sorted(default_prob.items(), key=lambda x: x[0])).values()

    prob = np.array(list(choice))
    prob = np.maximum(prob, 1e-10)
    prob = prob / prob.sum()
    return prob

def get_logloss(prob: np.array, item_pos: int):
    label = np.zeros_like(prob)
    if isinstance(item_pos, int):
        label[item_pos] = 1.
    else:
        label[ord(item_pos) - 65] = 1.
    logloss = -np.sum(label * np.log2(prob))
    return logloss

def process_prob_data(df: pd.DataFrame):
    df['prob'] = df.apply(lambda x: get_prob(x['choice'], x['choice_cnt']), axis=1)
    df['logloss'] = df.apply(lambda x: get_logloss(x['prob'], x['item_pos']), axis=1)
    df['entropy'] = df['prob'].apply(get_entropy)
    df['behavior'] = df['choice'].apply(lambda x: chr(sampling(x, 1) + 65))
    df['item_pos'] = df['item_pos'].apply(lambda x: chr(x + 65) if isinstance(x, int) else x)
    df['preference'] = df.apply(lambda x: x['item_pos'] == x['behavior'], axis=1)
    return df

def calculate_metrics(teacher: pd.DataFrame, student: pd.DataFrame) -> pd.DataFrame:
    df = teacher.groupby(level=[0,1]).first()

    df.loc[:, 'teacher_hit'] = teacher['preference'].groupby(level=0).mean()
    df.loc[:, 'student_hit'] = student['preference'].groupby(level=0).mean()
    df.loc[:, "diff_hit"] = df['student_hit'] - df['teacher_hit']

    df.loc[:, 'teacher_entropy'] = teacher['entropy'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_entropy'] = student['entropy'].groupby(level=[0,1]).mean()
    df.loc[:, 'teacher_uncertainty'] = teacher['uncertainty'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_uncertainty'] = student['uncertainty'].groupby(level=[0,1]).mean()
    df.loc[:, 'teacher_data_uncertainty'] = df.apply(lambda x: x["teacher_uncertainty"] - x["teacher_entropy"], axis=1)
    df.loc[:, 'student_data_uncertainty'] = df.apply(lambda x: x["student_uncertainty"] - x["student_entropy"], axis=1)
    
    df.loc[:, 'diff_all_uncertainty'] = df['student_uncertainty'] - df['teacher_uncertainty']
    df.loc[:, 'diff_model_uncertainty'] = df['student_entropy'] - df['teacher_entropy']
    df.loc[:, 'diff_data_uncertainty'] = (df['student_uncertainty'] - df['teacher_uncertainty']) - (df['student_entropy'] - df['teacher_entropy'])

    df.loc[:, 'teacher_logloss'] = teacher['logloss'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_logloss'] = student['logloss'].groupby(level=[0,1]).mean()
    return df


def load_content(project_path, dataset, variant):
    """
    Load data from JSON files.
    
    Args:
        project_path (str): Path to the project directory
        dataset (str): Name of the dataset
        variant (str): Type of data ('teacher' or 'student')
        
    Returns:
        pd.DataFrame: Loaded and joined dataframe
    """
    if os.path.exists(os.path.join(project_path, "datasets", f"{dataset}_train.jsonl")):
        samples = pd.read_json(os.path.join(project_path, "datasets", f"{dataset}_train.jsonl"), lines=True)
    else:
        raise FileNotFoundError(f"Dataset {dataset} not found in {project_path}")
    if os.path.exists(os.path.join(project_path, "probs", f"{dataset}_probs_{variant}.jsonl")):
        probs = pd.read_json(os.path.join(project_path, "probs", f"{dataset}_probs_{variant}.jsonl"), lines=True)
    else:
        raise FileNotFoundError(f"Probs {dataset} not found in {project_path}")
    
    # Join the dataframes
    df = probs.loc[:, ["decision_list", "choice"]].join(samples)
    
    # Process the dataframe
    df = df.explode(column=['choice', "decision_list"])
    df = df.dropna()
    
    return df