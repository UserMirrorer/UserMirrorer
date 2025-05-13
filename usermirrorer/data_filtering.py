"""
Final Filtering Module

This module provides functionality for processing, filtering, and preparing datasets for
user behavior mirroring tasks. It handles data preprocessing, metric calculation,
preference-based filtering, and final dataset creation.

The workflow:
1. Preprocess data from strong and weak variants
2. Calculate metrics for filtering
3. Sample behaviors based on specified criteria
4. Filter data according to configured parameters
5. Create the final preference dataset for training
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from usermirrorer.generator.template import texts_to_messages, convert_action_list
from usermirrorer.filter.utils import calculate_metrics, process_prob_data, get_entropy, load_content
from usermirrorer.filter.config import FilteringConfig

def data_preprocess(
        dataset: str,
        project_path: str,
    ):
    """
    Preprocess raw dataset by loading, aligning, and transforming strong and weak data variants.
    
    Args:
        dataset: The name of the dataset to preprocess
        project_path: The root path of the project containing the data
        
    Returns:
        tuple: (strong_data, weak_data) - Preprocessed dataframes with aligned indices
    """
    # Load strong and weak data
    strong = load_content(project_path, dataset, "strong")
    weak = load_content(project_path, dataset, "weak")
    
    # Ensure both datasets have the same indices for fair comparison
    common_indices = strong.index.intersection(weak.index)
    strong = strong.loc[common_indices]
    weak = weak.loc[common_indices]
    
    # Process probability data for both variants
    strong = process_prob_data(strong)
    weak = process_prob_data(weak)
    
    # Restructure indices for consistent access patterns
    strong = strong.reset_index().set_index("index", append=True).swaplevel(0, 1)
    weak = weak.reset_index().set_index("index", append=True).swaplevel(0, 1)

    # Filter non-preference data - keep only entries with valid preferences
    strong_valid_indices = strong["preference"].groupby(level=0).any()
    weak_valid_indices = weak["preference"].groupby(level=0).any()
    all_valid_indices = strong_valid_indices | weak_valid_indices
    all_valid_indices = all_valid_indices[all_valid_indices].index
    strong = strong.loc[all_valid_indices]
    weak = weak.loc[all_valid_indices]

    # Calculate uncertainty metrics based on probability distributions
    strong["uncertainty"] = strong["prob"].groupby(level=0).transform(lambda x: get_entropy(x.mean()))
    strong = strong.drop(columns=["prob", "choice", "decision_list", "text"]).reset_index().rename(columns={"index": "cid", "level_1": "tid"})

    weak["uncertainty"] = weak["prob"].groupby(level=0).transform(lambda x: get_entropy(x.mean()))
    weak = weak.drop(columns=["prob", "choice", "decision_list", "text"]).reset_index().rename(columns={"index": "cid", "level_1": "tid"})

    return strong, weak

def data_filtering(
    metrics: pd.DataFrame,
    filter_column: str,
    group_columns: list,
    filtered_num: int,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Filter data based on specified metrics and stratification criteria.
    
    This function implements stratified sampling to ensure balanced representation
    across different groups while prioritizing samples based on the filter column.
    
    Args:
        metrics: DataFrame containing metric values for filtering
        filter_column: Column name to use for filtering/sorting
        group_columns: Columns to use for stratification
        filtered_num: Maximum number of samples to select
        ascending: Sort order for the filter_column (default: False - highest values first)
        
    Returns:
        DataFrame: Filtered dataset with both chosen and rejected samples
    """
    filtered_metrics = metrics.copy()
    if filter_column == "random":
        # Random sampling without sorting
        filtered_metrics = filtered_metrics.sample(frac=1)
    else:
        # Sort based on the specified column
        filtered_metrics = filtered_metrics.sort_values(by=filter_column, ascending=ascending)
        
    # Calculate the total number of samples to select
    total_samples = min(filtered_num, len(filtered_metrics))
    
    # Create a default group if none specified
    if len(group_columns) == 0:
        filtered_metrics["group"] = "all"
        group_columns = ["group"]
        
    # Get group sizes for proportional allocation
    group_sizes = filtered_metrics.groupby(group_columns).size()
    total_original = sum(group_sizes)

    # Calculate proportional allocation for each group
    group_allocations = {}
    remaining = total_samples
    
    # Allocate samples proportionally to each group
    for group, size in group_sizes.items():
        # Calculate the proportion of samples to take from this group
        allocation = int(np.ceil((size / total_original) * total_samples))
        # Ensure we don't exceed the remaining samples or group size
        allocation = min(allocation, size, remaining)
        group_allocations[group if isinstance(group, tuple) else (group,)] = allocation
        remaining -= allocation

    # Select samples from each group according to allocations
    all_chosen = []
    all_rejected = []
    for group_name, group_df in filtered_metrics.groupby(group_columns):
        allocation = group_allocations.get(group_name if isinstance(group_name, tuple) else (group_name,), 0)
        # Split into preferred and rejected samples
        pref_df = group_df[group_df['preference']]
        rej_df = group_df[~group_df['preference']]
        if allocation > 0:
            # Take top samples based on sorting from each group
            chosen = pref_df.head(allocation)
            # Find matching rejected samples for the chosen ones
            rejected = rej_df.set_index(["dataset", "cid"]).loc[chosen.set_index(["dataset", "cid"]).index].reset_index()
            all_chosen.append(chosen.copy())
            all_rejected.append(rejected.copy())
    
    # Combine all selected samples
    sft_datasets = pd.concat([pd.concat(all_chosen), pd.concat(all_rejected)])
    return sft_datasets

def behavior_sampling(all_data: pd.DataFrame, metrics: pd.DataFrame, filter_column: str, sampling_column: str):
    """
    Sample behaviors from the dataset based on specified criteria.
    
    This function selects one behavior per context based on the sampling criteria,
    ensuring that each context has both preferred and non-preferred behaviors.
    
    Args:
        all_data: Combined dataset with all behaviors
        metrics: DataFrame with calculated metrics
        filter_column: Column used for filtering
        sampling_column: Column used for behavior sampling
    
    Returns:
        DataFrame: Sampled behaviors with associated metrics
    """
    # Extract relevant columns from the dataset
    sampled_df = all_data.reset_index(level=2).loc[metrics.set_index(["dataset", "cid"]).index, 
                                                   ["tid", "preference", "entropy", "logloss", "behavior", "variant"]]
    sampled_df = sampled_df.set_index('preference', append=True)
    
    # Sample behaviors based on specified criteria
    if sampling_column == "random":
        # Random sampling
        sampled_df = sampled_df.sample(frac=1).groupby(level=[0,1,2]).first()
    else:
        # Deterministic sampling based on column values
        sampled_df = sampled_df.sort_values(by=sampling_column, ascending=True).groupby(level=[0,1,2]).first()
    
    sampled_df = sampled_df.reset_index(level=2)
    
    # Keep only groups that have at least one True and one False preference (chosen/rejected pair)
    has_true = sampled_df['preference'].groupby(level=[0,1]).any()
    has_false = (~sampled_df['preference']).groupby(level=[0,1]).any()
    valid_indices = has_true & has_false
    sampled_df = sampled_df.loc[valid_indices]
    
    # Join with relevant metrics
    keep_columns = ["choice_cnt"]
    if filter_column in metrics.columns:
        keep_columns.append(filter_column)
    sampled_df = sampled_df.join(metrics.set_index(["dataset", "cid"]).loc[:, keep_columns])
    return sampled_df

def get_sample_content(filtered_df: pd.DataFrame, project_path: str):
    """
    Retrieve full content for the filtered samples.
    
    For each dataset, loads the original content and joins it with the filtered data.
    
    Args:
        filtered_df: DataFrame with filtered samples
        project_path: Path to the project directory
        
    Returns:
        DataFrame: Complete dataset with all required content for training
    """
    train_df = []
    for dataset, df in filtered_df.groupby("dataset"):
        # Load original content for each dataset
        strong_samples = load_content(project_path, dataset, "strong")
        weak_samples = load_content(project_path, dataset, "weak")
        
        # Ensure alignment between strong and weak samples
        common_indices = strong_samples.index.intersection(weak_samples.index)
        strong_samples = strong_samples.loc[common_indices].reset_index().set_index("index", append=True).swaplevel(0, 1)
        weak_samples = weak_samples.loc[common_indices].reset_index().set_index("index", append=True).swaplevel(0, 1)
        
        # Label samples by variant
        strong_samples["variant"] = "strong"
        weak_samples["variant"] = "weak"
        
        # Combine samples and restructure indices
        samples = pd.concat([strong_samples, weak_samples])
        samples = samples.reset_index().rename(columns={"index": "cid", "level_1": "tid"}).set_index(["cid", "variant", "tid"])
        
        # Join filtered data with full content
        train_df.append(df.set_index(["cid", "variant", "tid"]).join(
            samples.loc[:, ["text", "decision_list", "user_id", "item_id", "timestamp", "choice_cnt"]], 
            how="inner"))
    
    train_df = pd.concat(train_df)
    return train_df


def create_final_dataset(train_df: pd.DataFrame, mode: str = "decision", project_path: str = "", name: str = "UserMirrorer"):
    """
    Create the final preference dataset for training.
    
    Transforms the filtered data into a preference format with chosen and rejected pairs,
    then saves it to a JSONL file.
    
    Args:
        train_df: DataFrame with complete training data
        mode: Dataset creation mode (default: "decision")
        project_path: Path to the project directory
        name: Name prefix for the output file
        
    Returns:
        None: Saves the dataset to disk
    """
    # Define output path
    pref_df_path = os.path.join(project_path, "datasets", f"{name}_pref.jsonl")

    # Ensure consistent data types
    train_df["item_id"] = train_df["item_id"].astype(str)
    train_df["user_id"] = train_df["user_id"].astype(str)

    # Create messages format for training
    train_df['messages'] = train_df.apply(lambda x: texts_to_messages(
            convert_action_list(x['text']), assistant_prefix=f"{x['decision_list'].strip()}\nBehavior: [{x['behavior']}]",
            assistant_message=True
        ), axis=1)

    train_df = train_df.copy()

    # Reshape data for preference format (chosen/rejected pairs)
    train_df = train_df.reset_index(level=[1,2]).set_index(["dataset"], append=True)
    chosen = train_df[train_df["preference"]]
    rejected = train_df[~train_df["preference"]]
    train_df = chosen.join(rejected, lsuffix="_chosen", rsuffix="_rejected", how="inner")

    # Standardize column names and select final columns
    train_df = train_df.rename(columns={"user_id_chosen": "user_id", "item_id_chosen": "item_id", "timestamp_chosen": "timestamp"})
    train_df = train_df.loc[:, ["user_id", "item_id", "timestamp", "messages_chosen", "messages_rejected", "behavior_chosen", "behavior_rejected"]].reset_index(level=1)
    
    # Save to JSONL file
    train_df.to_json(pref_df_path, lines=True, orient="records", index=False)
    print("Final dataset size: ", train_df.shape[0])
    print("Pref dataset saved to ", pref_df_path)
    
if __name__ == "__main__":
    """
    Main execution entry point.
    
    Processes command line arguments, loads configuration, and executes the
    full data filtering pipeline.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="UserMirrorer", help="Configuration name to use")
    parser.add_argument("--project_path", type=str, help="Path to project directory")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ml-1m"], help="List of datasets to use")
    args = parser.parse_args()
    
    # Load configuration
    config = FilteringConfig(name=args.config_name, project_path=args.project_path, datasets=args.datasets)
    print(f"Filtering dataset with {config.name}...")

    # Initialize data containers
    strong_data = []
    weak_data = []

    # Process each dataset
    for ind, dataset in enumerate(config.datasets):
        strong, weak = data_preprocess(dataset, config.project_path)
        strong_data.append(strong)
        weak_data.append(weak)

    # Combine datasets
    strong_data = pd.concat(strong_data).set_index(["dataset", "cid", "tid"])
    weak_data = pd.concat(weak_data).set_index(["dataset", "cid", "tid"])

    print("Data Preprocessing Done! Total data size: ", strong_data.shape[0])

    # Calculate metrics for filtering
    metrics = calculate_metrics(strong_data, weak_data).reset_index()

    print("Calculate metrics Done!")

    # Label data variants
    strong_data["variant"] = "strong"
    weak_data["variant"] = "weak"

    # Combine data for behavior sampling
    all_data = pd.concat([strong_data, weak_data[~weak_data['preference']]])
    sampled_df = behavior_sampling(all_data, metrics, config.filter_column, config.sampling_column)

    print("Behavior Sampling Done!")

    # Apply filtering based on configuration
    print(f"Filtering dataset with {config.name}...")
    filtered_df= data_filtering(
        sampled_df.reset_index(),
        filter_column=config.filter_column,
        group_columns=config.group_columns,
        filtered_num=config.filtered_num,
        ascending=config.ascending    # Ascending: True -> Smallest diff_model_uncertainty; False -> Largest diff_model_uncertainty
    )

    # Select relevant columns for final dataset
    filtered_df = filtered_df.loc[:, ["dataset", "cid", "tid", "variant", "preference", "behavior"]]
    
    # Get complete content for filtered samples
    print(f"Get sample content ...")
    train_df = get_sample_content(filtered_df, config.project_path)
    
    # Create and save final dataset
    print(f"Creating final dataset ...")
    create_final_dataset(train_df, config.mode, config.project_path, config.name)
