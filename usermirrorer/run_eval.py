"""
Behavior Prediction Evaluation Module

This module provides functionality for evaluating the behavior prediction task using
an LLM-based classifier. It includes functions for reordering action lists, predicting
behavior, and running the evaluation process.

The workflow:
1. Load the dataset and convert action lists into text prompts
2. Predict behavior for each choice count
3. Compute accuracy of the predictions
4. Save the results to a JSONL file
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import pandas as pd
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from usermirrorer.generator.classifier import LLMClassifier
from usermirrorer.generator.template import texts_to_messages, convert_action_list
from usermirrorer.filter.utils import process_prob_data

def reorder_action_list(action_list: List[str], gt_idx: int, target_idx: int) -> List[str]:
    """Swap two elements in the action list at the given indices."""
    action_list[gt_idx], action_list[target_idx] = action_list[target_idx], action_list[gt_idx]
    return action_list

def predict(df: pd.DataFrame, model: LLMClassifier, choice_cnt: int, repeat_times: int = 1, mode: str = "direct", max_context_length: int = 6124) -> pd.DataFrame:
    """Generate predictions for DataFrame entries with a given number of choices."""
    # Filter DataFrame for the specified choice count
    df_slice = df[df["choice_cnt"] == choice_cnt]
    tokenizer = model.generator.model.get_tokenizer()
    df_slice = df_slice[df_slice['messages'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=True)).apply(len) <= max_context_length]
    df_slice = df_slice.copy()

    message_with_intent = df_slice['messages']
    if mode == "direct":
        output = model.run(
            message_with_intent.tolist(),
            choices=[f"{chr(65 + i)}" for i in range(choice_cnt)],
        )
        messages = [[""]] * len(output)
    else:
        output, messages = model.run_multiturn(
            message_with_intent.tolist(),
            choices=[f"{chr(65 + i)}" for i in range(choice_cnt)],
            stop_turn=["Behavior: ["],
            repeat_times=repeat_times
        )
    df.loc[df_slice.index, "choice"] = pd.Series(output, index=df_slice.index)
    df.loc[df_slice.index, "intent_list"] = pd.Series(messages, index=df_slice.index)
    return df


def run_eval(
        input_file: str,
        output_file: str,
        model: LLMClassifier,
        mode: str,
        repeat_times: int = 3
    ):
    """Load data, prepare prompts, run evaluation, and compute accuracy."""
    # Load dataset from JSON lines file
    df = pd.read_json(input_file, lines=True)
    # Convert action lists into text prompts
    df["text"] = df["text"].apply(convert_action_list)
    if mode == "direct":
        df['messages'] = df['text'].apply(lambda x: texts_to_messages(x, assistant_message=True, assistant_prefix="Behavior: [", system="direct"))
    else:
        df['messages'] = df['text'].apply(lambda x: texts_to_messages(x, assistant_message=True, assistant_prefix="Thought:\nStimulus:"))
    df["choice"] = df["choice_cnt"].apply(lambda x: [{f"{chr(65 + i)}": (1 / x) for i in range(x)}] * repeat_times)
    df["intent_list"] = df["messages"].apply(lambda x: [x] * repeat_times)

    for choice_cnt in tqdm(df["choice_cnt"].unique().tolist()):
        df = predict(df, model, choice_cnt, repeat_times, mode)

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    df.to_json(output_file, lines=True, orient="records", index=False)

    df = df.explode(["choice", "intent_list"])
    df = process_prob_data(df)
    accuracy = df.apply(lambda x: x['item_pos'] == x['behavior'], axis=1).groupby(level=0).mean().mean()
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":  
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    args.add_argument("--project_path", type=str)
    args.add_argument("--mode", type=str, default="default", choices=["default", "direct"])
    args.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use")
    args.add_argument("--repeat_times", type=int, default=3, help="The number of sampling times")
    args.add_argument("--input_file", type=str, default="")
    args.add_argument("--output_file", type=str, default="")
    args = args.parse_args()

    print(f"Loading model from {args.model_path}")
    model = LLMClassifier(model_path=args.model_path, gpu_device=args.gpu_ids)
    print(f"Running evaluation on {args.input_file}")
    run_eval(args.input_file, args.output_file, model, args.mode, args.repeat_times)
    print(f"Evaluation results saved to {args.output_file}")
