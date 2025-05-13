"""
Decision Generation Module

This module provides functionality for generating decision lists using an LLM. It includes
functions for creating batch files, running the LLM, and saving the results.

The workflow:
1. Create a batch file from the input dataset
2. Run the LLM to generate decision lists
3. Save the results to a JSONL file
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from usermirrorer.openai.openai import row_create
from usermirrorer.openai.batch import run_batch_openai, run_batch_vllm, run_batch_instant_api
from usermirrorer.generator.template import texts_to_messages, convert_action_list

from openai import OpenAI

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="MIND")
    args.add_argument("--model_path", type=str, default="gpt-4o-mini")
    args.add_argument("--project_path", type=str)
    args.add_argument("--version", type=str, default="the version of the decision-making process, e.g. 'strong' or 'weak'")
    args.add_argument("--gpu_ids", type=str, default="2", help="use comma-separated gpu ids for parallel processing")
    args = args.parse_args()

    input = os.path.join(args.project_path, "datasets", f"{args.dataset}_train.jsonl")
    batch_file = os.path.join(args.project_path, "decisions", f"{args.dataset}_batch.jsonl")
    if not os.path.exists(os.path.join(args.project_path, "decisions")):
        os.makedirs(os.path.join(args.project_path, "decisions"))
    df = pd.read_json(input, lines=True)

    def text_update(x, choice):
        """Append the chosen behavior as an update to the action_list field of a record."""
        x['action_list'] = x['action_list'] + f"\n\nBehavior: [{choice}]"
        return x

    df['messages'] = df['text'].apply(lambda x: texts_to_messages(convert_action_list(x)))
    df['batch_row'] = df.apply(lambda x: row_create(
        model=args.model_path,
        custom_id=f"{x['dataset']}-{x.name}",
        temperature=1.0,
        top_p=0.9,
        n=10,
        messages=x['messages'],
    ), axis=1)
    df['batch_row'].to_json(batch_file, lines=True, orient="records", index=False)
    chunk_num = len(args.gpu_ids.split(","))
    chunksize = 1 + df.shape[0] // chunk_num

    if "gpt" in args.model_path or "o1" in args.model_path or "deepseek" in args.model_path or "qwen" in args.model_path:
        client = OpenAI()
        run_batch_openai(
            input_file=batch_file,
            output_file=os.path.join(args.project_path, "decisions", f"{args.dataset}_decisions_{args.version}.jsonl"),
            model=args.model_path,
            client=client
        )
    else:
        run_batch_vllm(
            input_file=batch_file,
            output_file=os.path.join(args.project_path, "decisions", f"{args.dataset}_decisions_{args.version}.jsonl"),
            model=args.model_path,
            client=None,
            **{"data_parallel_size": chunk_num, "chunksize": chunksize, "gpu_ids": args.gpu_ids}
        )

    os.remove(batch_file)