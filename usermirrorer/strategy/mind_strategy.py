import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Callable, List
from collections import ChainMap

from ..formatter.strategy import DataStrategy
from ..formatter.mapping import MappingStrategy, do_nothing
from ..formatter.utils import process_timestamp, format_time_diff
from ..sampler.sampler import SampleFormulator, time_filtering, interaction_filtering
from ..formatter.formatter import DataFormatter


template_profile = """"""

template_action_list = """{title} (Category: {category})
"""

template_history = """{title} (Category: {category}) {{ Viewed {time_diff} }}
"""

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
        sampled_inter["timestamp"] - process_timestamp(float(item["item_features"]["time"]))
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
    features = []
    try:
        features.append({"time_diff": format_time_diff( 
            sampled_inter["timestamp"] - process_timestamp(item["timestamp"])
        )})
    except:
        features.append({"time_diff": "earlier"})
    return features

class MINDDataStrategy(DataStrategy):
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
            "action_list": feature_action_list,
            "history": feature_history,
            "profile": do_nothing
        }
    
class MINDMappingStrategy(MappingStrategy):
    def map_history(
        self,
        inter: dict,
        user_df: pd.DataFrame,
        item_df: pd.DataFrame,
        interaction_df: pd.DataFrame,
        feature_func: Callable[[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame], dict] = do_nothing,
        random_truncate_history: bool = True,
    ) -> List[dict]:
        """Maps features from user's interaction history."""
        user_history = interaction_df.loc[[inter["user_id"]]]
        user_history = user_history.loc[user_history["timestamp"] < inter["timestamp"], ["item_id", "timestamp", "behavior_features"]]
        user_history = user_history.sort_values(by="timestamp", ascending=True)
        static_history = user_df.loc[inter["user_id"], "user_features"]["history"][-10:]
        static_history = pd.DataFrame({
            "user_id": [inter["user_id"]] * len(static_history),
            "item_id": static_history,
            "timestamp": [float("-inf")] * len(static_history),
            "behavior_features": [{}] * len(static_history)
        })
        user_history = pd.concat([static_history, user_history], ignore_index=True)
        if random_truncate_history:
            user_history = user_history.iloc[-random.randint(2, self.max_history_len):]
        else:
            user_history = user_history.iloc[-self.max_history_len:]
        user_history = user_history.join(item_df, on="item_id", how="left")
        if user_history.empty:
            return []

        features = user_history.apply(
            lambda x: dict(ChainMap(*([x["item_description"]] + \
                                      feature_func(x, inter, user_df, item_df, interaction_df)))
            ), axis=1
        )
        return features.to_list()

# ---------------------Sampling---------------------
def init_sampler(sf: SampleFormulator, df: DataFormatter):
    sf.add_scoring_method("collaborative", interaction_index=True)
    sf.add_scoring_method("content-based", interaction_index=False)
    sf.add_filtering_method(time_filtering)
    sf.add_filtering_method(interaction_filtering)
