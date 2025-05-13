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



template_profile = """## User Profile
A user who has purchased {products} games
"""

template_action_list = """Title: {app_name} [{genres}]
Developed by {developer} | Published by {publisher}
Released: {release_date} 
Tags: {tags}
Specifications: {specs}
Player Reviews: {sentiment}
Price: {price}
"""

template_history = """A game purchased {time_diff}
Title: {app_name} [{genres}]
Developed by {developer} | Published by {publisher}
Released: {release_date} 
Tags: {tags}
Specifications: {specs}
Player Reviews: {sentiment}
Price: {price}
My Behavior: Played {hours_played} hours | Review: {shorten_review_text}
"""


def truncate_text(text: str, max_length: int, by_words: bool = False) -> str:
    """Truncates text to specified length, adding ellipsis if truncated.
    
    Args:
        text: String to truncate
        max_length: Maximum length (in words or characters)
        by_words: If True, truncate by word count. If False, truncate by character count.
    
    Returns:
        Truncated string with ellipsis if needed
    """
    if not text:
        return text
        
    if by_words:
        words = text.split()
        if len(words) <= max_length:
            return text
        return ' '.join(words[:max_length]) + '...'
    else:
        if len(text) <= max_length:
            return text
        return text[:max_length] + '...'

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
    return []

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
    features = [{"hours_played": str(item["behavior_features"]["hours_played"])},
                {"shorten_review_text": truncate_text(item["behavior_features"]["review_text"], 50, by_words=True)}]
    try:
        features.append({"time_diff": format_time_diff( 
            process_timestamp(sampled_inter["timestamp"]) - process_timestamp(item["timestamp"])
        )})
    except:
        features.append({"time_diff": "earlier"})
    return features


# %%
class SteamDataStrategy(DataStrategy):
    def _get_default_templates(self) -> Dict[str, str]:
        """Return default templates for steam dataset."""
        return {
            "action_list": template_action_list,
            "history": template_history,
            "profile": template_profile
        }

    def _get_default_feature_funcs(self) -> Dict[str, Callable]:
        """Return default feature functions for steam dataset."""
        return {
            "action_list": feature_action_list,
            "history": feature_history,
            "profile": do_nothing
        }

# ---------------------Sampling---------------------

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
    sf.add_scoring_method("collaborative", interaction_index=True)
    sf.add_scoring_method("content-based", interaction_index=False)
    sf.add_filtering_method(time_filtering)
    sf.add_filtering_method(interaction_filtering)
