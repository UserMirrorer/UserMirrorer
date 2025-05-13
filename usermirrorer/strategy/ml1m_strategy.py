# %%
import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Callable, List
from datetime import datetime
from collections import ChainMap

from ..formatter.strategy import DataStrategy
from ..formatter.mapping import MappingStrategy, do_nothing
from ..formatter.utils import process_timestamp, format_time_diff
from ..sampler.sampler import SampleFormulator, interaction_filtering
from ..formatter.formatter import DataFormatter




template_profile = """## User Profile
Gender: {gender}
Age: {age}
Occupation: {occupation}
Location: {location}
"""

template_action_list = """{new_release}{title} - {genres} 
"""

template_history = """A movie viewed {time_diff}
{title} {genres} 
My Behavior:
Rating: {rating}/5.0
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
    return [
        {
            "new_release": "[New Release]" if (sampled_inter["timestamp"] - datetime.strptime(item["item_features"]["year"], '%Y')).days <= 365 else "",
        }
    ]

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
    features = [{"rating": str(item['behavior_features']['rating'])}]
    try:
        features.append({"time_diff": format_time_diff( 
            sampled_inter["timestamp"] - process_timestamp(item["timestamp"])
        )})
    except:
        features.append({"time_diff": "earlier"})
    return features


# %%
class ML1MDataStrategy(DataStrategy):
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

from datetime import datetime
def time_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items based on timestamp constraints.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with future items removed
    """
    time = sf.item_df.set_index('item_id')['item_features'].apply(lambda x: datetime.strptime(x['year'], '%Y'))
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if time.loc[i] <= x['timestamp']],
        axis=1
    )
    return sampled_inter


def init_sampler(sf: SampleFormulator, df: DataFormatter):
    sf.add_scoring_method("collaborative", interaction_index=True)
    sf.add_scoring_method("content-based", interaction_index=False)
    sf.add_filtering_method(time_filtering)
    sf.add_filtering_method(interaction_filtering)

