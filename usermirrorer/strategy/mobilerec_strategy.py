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

template_action_list = """{app_name} ({avg_rating}/5.0 - {num_reviews} reviews)
Category: {app_category}
Developer: {developer_name}
Price: {price}

"""

template_history = """{app_name} ({avg_rating}/5.0 - {num_reviews} reviews)
Category: {app_category}
Developer: {developer_name}
Price: {price}
Description: {short_description}
My Rating: {rating}
My Reviews: {review}

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


def feature_action_list(
    item: dict,
    sampled_inter: dict,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> List[dict]:
    """Extract features for action list items in MIND dataset.
    
    Args:
        item: Dictionary containing item information
        sampled_inter: Dictionary containing interaction information
        user_df: DataFrame containing user information
        item_df: DataFrame containing item information
        interaction_df: DataFrame containing interaction information
    
    Returns:
        List of dictionaries containing extracted features
    """
    return [{"time_diff": format_time_diff(
        sampled_inter["timestamp"] - process_timestamp(item["item_features"]["time"])
    )}]

def feature_history(
    item: dict,
    sampled_inter: dict,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> List[dict]:
    """Extract features for history items in MIND dataset.
    
    Args:
        item: Dictionary containing item information
        sampled_inter: Dictionary containing interaction information
        user_df: DataFrame containing user information
        item_df: DataFrame containing item information
        interaction_df: DataFrame containing interaction information
    
    Returns:
        List of dictionaries containing extracted features
    """
    features = [
        {"rating": item["behavior_features"]["rating"]},
        {"review": item["behavior_features"]["review"]},
        {"short_description": truncate_text(item["item_description"]["description"], 20, by_words=True)},
    ]
    try:
        features.append({"time_diff": format_time_diff( 
            sampled_inter["timestamp"] - process_timestamp(item["timestamp"])
        )})
    except:
        features.append({"time_diff": "earlier"})

    return features

class MobilerecDataStrategy(DataStrategy):
    def _get_default_templates(self) -> Dict[str, str]:
        """Return default templates for MIND dataset."""
        return {
            "action_list": template_action_list,
            "history": template_history,
            "profile": template_profile
        }

    def _get_default_feature_funcs(self) -> Dict[str, Callable]:
        """Return default feature functions for MIND dataset."""
        return {
            "action_list": do_nothing,
            "history": feature_history,
            "profile": do_nothing
        }
    
# ---------------------Sampling---------------------


def init_sampler(sf: SampleFormulator, df: DataFormatter):
    sf.add_scoring_method("collaborative", interaction_index=True)
    sf.add_scoring_method("content-based", interaction_index=False)
    sf.add_filtering_method(interaction_filtering)

