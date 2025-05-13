# %%
# %%
import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Callable, List
from collections import ChainMap

from ..formatter.strategy import DataStrategy
from ..formatter.mapping import MappingStrategy, do_nothing
from ..formatter.utils import process_timestamp, format_time_diff
from ..sampler.sampler import SampleFormulator, interaction_filtering
from ..formatter.formatter import DataFormatter


template_profile = """"""

template_action_list = """Artist name: {name} ({listened_friends})
Tags: {tags}
"""

template_history = """An artist listened {time_diff}
Artist name: {name} 
Tags: {tags}
My Behavior: Tagging with {tagValue}
"""

# %%
def feature_action_list(
    item: dict,
    sampled_inter: dict,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> List[dict]:
    """Extract features for action list items in goodreads dataset.
    
    Args:
        item: Dictionary containing item information
        sampled_inter: Dictionary containing interaction information
        user_df: DataFrame containing user information
        item_df: DataFrame containing item information
        interaction_df: DataFrame containing interaction information
    
    Returns:
        List of dictionaries containing extracted features
    """
    try:
        user_features = user_df.loc[sampled_inter["user_id"]]["user_features"]
        friend_num = int(user_features["social_rel"][item.name]) if "social_rel" in user_features else 0
        return [
            {"listened_friends": f"{friend_num} friend(s) listened" if friend_num > 0 else ""}
        ]
    except:
        return [{"listened_friends": ""}]

def feature_history(
    item: dict,
    sampled_inter: dict,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> List[dict]:
    """Extract features for history items in goodreads dataset.
    
    Args:
        item: Dictionary containing item information
        sampled_inter: Dictionary containing interaction information
        user_df: DataFrame containing user information
        item_df: DataFrame containing item information
        interaction_df: DataFrame containing interaction information
    
    Returns:
        List of dictionaries containing extracted features
    """
    features = []
    
    # ensure tagValue is not None 
    if "behavior_features" in item and "tagValue" in item["behavior_features"]:
        tag_value = item["behavior_features"]["tagValue"]
        # 将tagValue统一转换为字符串形式
        if isinstance(tag_value, list):
            # 如果是列表,将元素用逗号连接
            tag_str = ", ".join(str(t) for t in tag_value if pd.notna(t))
            features.append({"tagValue": tag_str if tag_str else "No tags"})
        elif isinstance(tag_value, str):
            # 如果已经是字符串,直接使用
            features.append({"tagValue": tag_value if pd.notna(tag_value) else "No tags"})
        else:
            # 其他类型转换为字符串
            features.append({"tagValue": str(tag_value) if pd.notna(tag_value) else "No tags"})
    else:
        features.append({"tagValue": "No tags"})

    # process time_diff
    try:
        time_diff = format_time_diff(
            process_timestamp(sampled_inter["timestamp"]) - process_timestamp(item["timestamp"])
        )
        features.append({"time_diff": time_diff})
    except:
        features.append({"time_diff": "earlier"})
    
    return features


# %%
class LastFMDataStrategy(DataStrategy):
    def _get_default_templates(self) -> Dict[str, str]:
        """Return default templates for goodreads dataset."""
        return {
            "action_list": template_action_list,
            "history": template_history,
            "profile": template_profile
        }

    def _get_default_feature_funcs(self) -> Dict[str, Callable]:
        """Return default feature functions for goodreads dataset."""
        return {
            "action_list": feature_action_list,
            "history": feature_history,
            "profile": do_nothing
        }

# ---------------------Sampling---------------------


import torch

def friends_id_map(friend_ids: List[int], user_id_map: Dict[int, int]) -> List[int]:
    friend_uids = []
    for friend_id in friend_ids:
        if friend_id in user_id_map:
            friend_uids.append(user_id_map[friend_id])
    return friend_uids


def construct_social_scoring_matrix(sf: SampleFormulator):   
    friend_uids = sf.user_df['user_features'].apply(lambda x: friends_id_map(x['friends'], sf.user_id_map))
    indices = torch.tensor(friend_uids.explode().reset_index().dropna().to_numpy().astype(int))
    social_graph = torch.sparse_coo_tensor(
        indices=indices.T,
        values=torch.ones(indices.shape[0]),
        size=(sf.user_df.index.nunique(), sf.user_df.index.nunique())
    ).coalesce()
    return torch.sparse.mm(social_graph, sf.user_item_interaction_matrix)

def social_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on content similarity using item embeddings.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Content-based similarity scores for each item
    """
    user_ids = sf.interaction_df['uid'].loc[sampled_inter_ids].to_numpy()
    inter_mat = sf.social_scoring_matrix.index_select(0, torch.tensor(user_ids).to(torch.long)).to_dense()
    return inter_mat

# %%
def add_social_relations_to_user_features(sf: SampleFormulator, df: DataFormatter):
    """
    Convert sparse social scoring matrix to user features and update DataFormatter.
    
    Args:
        sf: SampleFormulator instance with social_scoring_matrix
        df: DataFormatter instance to update
    """
    indices = sf.social_scoring_matrix.indices()
    values = sf.social_scoring_matrix.values()
    rows = []

    reverse_id_map = {v: k for k, v in sf.item_id_map.items()}

    # Group by row index
    for row_idx in range(sf.social_scoring_matrix.size(0)):
        row_mask = indices[0] == row_idx
        row_cols = indices[1][row_mask]
        row_vals = values[row_mask]
        
        # Create dictionary for this row
        row_dict = {reverse_id_map[col.item()]: val.item() for col, val in zip(row_cols, row_vals)}
        rows.append(row_dict)

    # Convert to DataFrame
    sf.user_df = sf.user_df.join(pd.Series(rows, name="social_rel"))
    sf.user_df['user_features'] = sf.user_df.apply(
        lambda x: {**x['user_features'], **{"social_rel": x['social_rel']}} if isinstance(x['social_rel'], dict) else x['user_features'], 
        axis=1
    )
    sf.user_df = sf.user_df.drop(columns=['social_rel'])
    df.user_df = sf.user_df.set_index('user_id')
    df.formatting_params = (df.user_df, df.item_df, df.interaction_df)

# %%
from datetime import datetime
def get_release_date(x: dict) -> datetime:
    try:
        return datetime.strptime(x['release_date'], '%Y-%m-%d')
    except:
        return datetime.strptime('1970-01-01', '%Y-%m-%d')

def time_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items based on timestamp constraints.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with future items removed
    """
    time = sf.item_df.set_index('item_id')['item_features'].apply(get_release_date)
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if time[i] <= x['timestamp']],
        axis=1
    )
    return sampled_inter

def init_sampler(sf: SampleFormulator, df: DataFormatter):
    sf.social_scoring_matrix = construct_social_scoring_matrix(sf)
    sf.add_scoring_method("collaborative", interaction_index=True)
    sf.add_scoring_method(social_scoring, interaction_index=True)
    add_social_relations_to_user_features(sf, df)

    sf.add_scoring_method("content-based", interaction_index=False)
    sf.add_filtering_method(time_filtering)
    sf.add_filtering_method(interaction_filtering)