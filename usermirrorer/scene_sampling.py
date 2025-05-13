# This script processes various recommendation dataset formats into a standardized format for LLM input
# It handles multiple dataset types and converts them using appropriate strategies

import os
import sys
import argparse
import pandas as pd
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from usermirrorer.formatter.formatter import DataFormatter
from usermirrorer.formatter.mapping import MappingStrategy
from usermirrorer.sampler.sampler import SampleFormulator

def get_data_strategy(dataset_name):
    """
    Select and return the appropriate data processing strategy based on dataset name.
    Each dataset has its own strategy class to handle its specific format and requirements.
    
    Args:
        dataset_name (str): Name of the dataset to process
        
    Returns:
        class: The dataset-specific strategy class
        
    Raises:
        NotImplementedError: If the dataset is not supported
    """
    if dataset_name == "ml-1m":
        from usermirrorer.strategy.ml1m_strategy import ML1MDataStrategy, init_sampler
        return ML1MDataStrategy, init_sampler
    elif dataset_name == "steam":
        from usermirrorer.strategy.steam_strategy import SteamDataStrategy, init_sampler
        return SteamDataStrategy, init_sampler
    elif dataset_name == "mobilerec":
        from usermirrorer.strategy.mobilerec_strategy import MobilerecDataStrategy, init_sampler
        return MobilerecDataStrategy, init_sampler
    elif dataset_name == "MIND":
        from usermirrorer.strategy.mind_strategy import MINDDataStrategy, init_sampler
        return MINDDataStrategy, init_sampler
    elif dataset_name == "KuaiRec2":
        from usermirrorer.strategy.kuairec_strategy import KuairecDataStrategy, init_sampler
        return KuairecDataStrategy, init_sampler
    elif dataset_name == "LastFM":
        from usermirrorer.strategy.lastfm_strategy import LastFMDataStrategy, init_sampler
        return LastFMDataStrategy, init_sampler
    elif dataset_name == "goodreads":
        from usermirrorer.strategy.goodreads_strategy import GoodreadsDataStrategy, init_sampler
        return GoodreadsDataStrategy, init_sampler
    elif "Amazon" in dataset_name:
        from usermirrorer.strategy.amazon_strategy import AmazonDataStrategy, init_sampler
        return AmazonDataStrategy, init_sampler
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

def get_mapping_strategy(dataset_name):
    """
    Get the appropriate mapping strategy for converting dataset-specific formats.
    Currently handles MIND datasets specially, uses default mapping for others.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        class: The mapping strategy class to use
    """
    if dataset_name == "MIND":
        from usermirrorer.strategy.mind_strategy import MINDMappingStrategy
        return MINDMappingStrategy
    else:
        return MappingStrategy

if __name__ == "__main__":
    # Set up command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="ml-1m", help="Name of the dataset to process")
    args.add_argument("--project_path", type=str, help="Path to the project directory")
    args.add_argument("--max_exposure_length", type=int, default=5, help="Max exposure length")
    args.add_argument("--min_exposure_length", type=int, default=1, help="Min exposure length")
    args.add_argument("--sample_nums", type=int, default=256, help="Sample numbers")
    args.add_argument("--eval_set", action="store_true", help="Whether to sample training set or eval set")
    args.add_argument("--disable_collaborative_embedding", action="store_true", help="Whether to not construct collaborative embedding")
    args.add_argument("--disable_content_embedding", action="store_true", help="Whether to not construct content embedding")
    args.add_argument("--embedding_model_path", type=str, help="Path to the LLM embedding model")
    args.add_argument("--seed", type=int, default=0, help="Random seed")
    args.add_argument("--length_filtering", type=int, default=-1, help="The maximum length of the context, set -1 to disable length filtering")
    args.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Path to the tokenizer, only used when length filtering is enabled")

    args = args.parse_args()
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    # Initialize data processing strategies
    data_strategy_class, init_sampler = get_data_strategy(args.dataset)
    mapping_strategy_class = get_mapping_strategy(args.dataset)
    print(f"Data strategy class: {data_strategy_class}")
    print(f"Mapping strategy class: {mapping_strategy_class}")
    # Create formatter with appropriate strategies
    data_formatter = DataFormatter(
        ds=data_strategy_class(dataset_name=args.dataset, dataset_path=args.project_path),
        mp=mapping_strategy_class()
    )
    print(f"Data formatter created.")
    sample_formulator = SampleFormulator(
        ds=data_strategy_class(dataset_name=args.dataset, dataset_path=args.project_path),
        item_num_range=(args.min_exposure_length, args.max_exposure_length),
    )
    print(f"Sample formulator created.")
    init_sampler(sample_formulator, data_formatter)
    print(f"Sampler initialized.")
    if sample_formulator.item_collaborative_embedding is None and not args.disable_collaborative_embedding:
        sample_formulator.construct_collaborative_features()
    if sample_formulator.item_content_embedding is None and not args.disable_content_embedding:
        import vllm
        llm_embedding = vllm.LLM(model=args.embedding_model_path, task="embedding")
        sample_formulator.construct_content_features(llm_embedding)
    print(f"Content embedding constructed.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    if not args.eval_set and os.path.exists(os.path.join(args.project_path, "datasets", f"{args.dataset}_eval.jsonl")):
        cand_user_ids = pd.read_json(os.path.join(args.project_path, "datasets", f"{args.dataset}_eval.jsonl"), orient="records", lines=True)
        cand_user_ids = cand_user_ids["user_id"].unique().tolist()
        if args.dataset == "MIND":
            sampled_dataframe = data_formatter.interaction_df
            sampled_dataframe = sampled_dataframe[sampled_dataframe["user_id"].isin(cand_user_ids)]
            sampled_dataframe["impression_list"] = sampled_dataframe["behavior_features"].progress_apply(lambda x: x["impression_list"])
            sampled_dataframe["item_pos"] = sampled_dataframe["behavior_features"].progress_apply(lambda x: x["item_pos"])
            sampled_dataframe["choice_cnt"] = sampled_dataframe["behavior_features"].progress_apply(lambda x: len(x["impression_list"]))
            sampled_dataframe = sampled_dataframe[sampled_dataframe["choice_cnt"] <= args.max_exposure_length]
            sampled_dataframe = sampled_dataframe[sampled_dataframe["choice_cnt"] >= args.min_exposure_length]
            sampled_dataframe = sampled_dataframe.sample(n=args.sample_nums)
        else:
            sample_dataframe = sample_formulator.sampling(n_samples=args.sample_nums, user_ids=cand_user_ids)
    else:
        sample_dataframe = sample_formulator.sampling(n_samples=args.sample_nums)
    print(f"Sampling done. Total samples: {len(sample_dataframe)}")
    print(f"Getting all details of sampled data. (This may take a while...)")
    sampled_inter = data_formatter.get_all_details(sample_dataframe)
    print(f"All details of sampled data got.")
    sampled_inter["dataset"] = data_formatter.ds.dataset_name

    if args.length_filtering != -1:
        from usermirrorer.filter.length_filter import length_filtering
        sampled_inter = length_filtering(
            data=sampled_inter,
            tokenizer_path=args.tokenizer_path,
            length=args.length_filtering
        )

    if not os.path.exists(os.path.join(args.project_path, "datasets")):
        os.makedirs(os.path.join(args.project_path, "datasets"))
    if not args.eval_set:
        sampled_inter.to_json(os.path.join(args.project_path, "datasets", f"{args.dataset}_train.jsonl"), orient="records", lines=True)
        print(f"Training set saved to {os.path.join(args.project_path, 'datasets', f'{args.dataset}_train.jsonl')}")
    else:
        sampled_inter.to_json(os.path.join(args.project_path, "datasets", f"{args.dataset}_eval.jsonl"), orient="records", lines=True)
        print(f"Test set saved to {os.path.join(args.project_path, 'datasets', f'{args.dataset}_eval.jsonl')}")
