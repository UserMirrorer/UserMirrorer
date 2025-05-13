import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from .strategy import DataStrategy
from .mapping import MappingStrategy


tqdm.pandas()

def get_interaction_order(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives the chronological order of interactions for each user based on timestamps.
    
    Args:
        interaction_df (pd.DataFrame): DataFrame containing user interactions with timestamps
        
    Returns:
        pd.DataFrame: Original DataFrame with an additional 'order_id' column indicating 
                     the chronological order of interactions per user
    
    Steps:
    1. Reset index to create interaction_id
    2. Sort values by timestamp
    3. Group by user_id and apply reset_index to get order within each group
    4. Join the order_id back to the original DataFrame
    """
    sorted_id = interaction_df.reset_index(
        names="interaction_id").sort_values(
            by="timestamp", ascending=True).groupby(
                "user_id")["interaction_id"].apply(
                    lambda x: x.reset_index(drop=True))
    sorted_id.index = sorted_id.index.set_names(["user_id", "order_id"])
    return interaction_df.join(sorted_id.reset_index().set_index("interaction_id").loc[:, "order_id"])


class DataFormatter:
    """
    A class to handle the formatting of user-item interaction data.
    
    This class loads and processes user, item, and interaction data according to 
    a specified formatting strategy. It provides methods to sample interactions
    and format them into structured text representations.
    
    Attributes:
        ds (DataStrategy): Strategy object containing dataset configuration and formatting rules
        user_df (pd.DataFrame): DataFrame containing user features
        item_df (pd.DataFrame): DataFrame containing item features
        interaction_df (pd.DataFrame): DataFrame containing user-item interactions
        feature_funcs (dict): Dictionary of functions to extract different feature types
        templates (dict): Dictionary of templates for text formatting
        action_type (str): Type of action/interaction being processed
    """
    
    def __init__(
            self,
            ds: DataStrategy,
            mp: MappingStrategy
        ):
        """
        Initialize the DataFormatter with a specific data strategy.
        
        Args:
            ds (DataStrategy): Strategy object containing dataset configuration
        """
        self.ds = ds
        self.mp = mp
        self._load_data()
        self._load_strategy()
        self.formatting_params = (self.user_df, self.item_df, self.interaction_df)

    def _load_data(self):
        """
        Load user, item, and interaction data from JSON files.
        
        Reads three types of data:
        - User features from *_user_feature.jsonl
        - Item features from *_item_feature.jsonl
        - Interaction data from *_interaction.jsonl
        
        The interaction data is processed to include chronological ordering.
        """
        self.user_df = pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_user_feature.jsonl"), lines=True).set_index("user_id")
        self.item_df = pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_item_feature.jsonl"), lines=True).set_index("item_id")
        self.interaction_df = get_interaction_order(pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_interaction.jsonl"), lines=True)).set_index("user_id")

    def _load_strategy(self):
        """
        Load formatting strategy parameters from the DataStrategy object.
        
        Sets up:
        - Feature extraction functions
        - Text templates
        - Action type for interactions
        """
        self.feature_funcs = self.ds._get_default_feature_funcs()
        self.templates = self.ds._get_default_templates()
        self.action_type = self.ds.action_type

    
    def get_formatted_text(self, sampled_inter: pd.DataFrame):
        """
        Convert sampled interactions into formatted text representations.
        
        Args:
            sampled_inter (pd.DataFrame): DataFrame containing sampled interactions
            
        Returns:
            pd.Series: Series containing dictionaries with formatted text for:
                      - profile: User profile information
                      - action_list: List of user actions/interactions
                      - history: Historical interaction data
                      
        Each text component is generated using the corresponding feature function
        and template from the formatting strategy.
        """
        return sampled_inter.progress_apply(
            lambda inter: {
                "profile": self.mp.get_text(self.mp.get_feature(
                    inter, *self.formatting_params,
                    field="profile",
                    feature_func=self.feature_funcs["profile"]
                ), template=self.templates["profile"]),
                "action_list": self.mp.get_text(self.mp.get_feature(
                    inter, *self.formatting_params,
                    field="action_list",
                    action_type=self.action_type,
                    feature_func=self.feature_funcs["action_list"]
                ), template=self.templates["action_list"], enum=False, combine=False),
                "history": self.mp.get_text(self.mp.get_feature(
                    inter, *self.formatting_params,
                    field="history",
                    feature_func=self.feature_funcs["history"]
                ), template=self.templates["history"])
            }, axis=1
        )
    
    def get_all_details(self, sampled_inter: pd.DataFrame):
        sampled_inter["text"] = self.get_formatted_text(sampled_inter)
        sampled_inter["choice_cnt"] = sampled_inter["impression_list"].apply(len)
        return sampled_inter.loc[:, ["text", "user_id", "item_id", "timestamp", "item_pos", "choice_cnt"]]
