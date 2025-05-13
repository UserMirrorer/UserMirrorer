import os
import random
import pandas as pd
from collections import ChainMap
from typing import List, Literal, Callable, Dict, Any
from abc import ABC, abstractmethod

random.seed(0)

def do_nothing(
    item: dict,
    sampled_inter: dict,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> List[dict]:
    """
    A placeholder function that returns an empty list. Used as default feature_func.
    
    Args:
        item: Dictionary containing item information
        sampled_inter: Dictionary containing interaction information
        user_df: DataFrame containing user information
        item_df: DataFrame containing item information
        interaction_df: DataFrame containing interaction information
    
    Returns:
        Empty list
    """
    return []

class MappingStrategy(ABC):
    def __init__(
        self,
        max_history_len: int = 10,
    ):
        self.max_history_len = max_history_len

    @staticmethod
    def get_text(
        features: dict | List[dict],
        enum: bool = False,
        template: str = "",
        list_sep: str = "\n",
        combine: bool = True,
    ) -> str:
        """
        Convert features dictionary or list of dictionaries into formatted text using a template.
        
        Args:
            features (dict | List[dict]): Features to format, either a single dictionary or list of dictionaries
            enum (bool, optional): Whether to enumerate list items with letters [A], [B], etc. Defaults to False.
            template (str, optional): Template string to format features with. Defaults to empty string.
            list_sep (str, optional): Separator string for list items. Defaults to newline.
            
        Returns:
            str: Formatted text string with features inserted into template
            
        Raises:
            ValueError: If features is not a dict or list
        """
        if isinstance(features, dict):
            return template.format(**features)
        elif isinstance(features, list):
            if enum:
                elements = [f"[{chr(idx + 65)}] " + template.format(**feature) for idx, feature in enumerate(features)]
            else:
                elements = [template.format(**feature) for feature in features]
            if combine:
                return list_sep.join(elements)
            else:
                return elements
        else:
            raise ValueError(f"Invalid features type: {type(features)}")

    def get_feature(
        self,
        inter: pd.DataFrame,
        user_df: pd.DataFrame,
        item_df: pd.DataFrame,
        interaction_df: pd.DataFrame,
        field: Literal["profile", "history", "action_list"] = "profile",
        action_type: Literal["obj", "text"] = "obj",
        feature_func: Callable[[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame], dict] = do_nothing,
    ) -> dict | List[dict]:
        """
        Main function to get features based on specified field and action type.
        
        Args:
            inter: DataFrame row containing interaction information
            user_df: DataFrame containing user information
            item_df: DataFrame containing item information
            interaction_df: DataFrame containing interaction information
            field: Type of field to map ('profile', 'history', or 'action_list')
            action_type: Type of action ('obj' or 'text')
            feature_func: Optional function to extract additional features
        
        Returns:
            Dictionary or List of dictionaries containing mapped features
        """
        if field == "profile":
            mapping_func = self.map_profile
        elif field == "history":
            mapping_func = self.map_history
        elif field == "action_list":
            mapping_func = self.map_action_list
        
        if action_type == "text":
            return inter[field]
        return mapping_func(inter, user_df, item_df, interaction_df, feature_func)

    def map_profile(
        self,
        inter: dict,
        user_df: pd.DataFrame,
        item_df: pd.DataFrame,
        interaction_df: pd.DataFrame,
        feature_func: Callable[[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame], dict] = do_nothing,
    ) -> dict:
        """Maps user profile features."""
        user = user_df.loc[inter["user_id"]]
        features = dict(ChainMap(*([user["user_description"]] + \
                            feature_func(user, inter, user_df, item_df, interaction_df)))
        )

        return features

    def map_action_list(
        self,
        inter: dict,
        user_df: pd.DataFrame,
        item_df: pd.DataFrame,
        interaction_df: pd.DataFrame,
        feature_func: Callable[[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame], dict] = do_nothing,
    ) -> List[dict]:
        """Maps features for a list of actions."""
        items = item_df.loc[inter['impression_list']]
        features = items.apply(
            lambda x: dict(ChainMap(*([x["item_description"]] + \
                                      feature_func(x, inter, user_df, item_df, interaction_df)))
            ), axis=1
        )
        return features.to_list()

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
