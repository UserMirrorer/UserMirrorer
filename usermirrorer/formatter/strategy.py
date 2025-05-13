import pandas as pd
from typing import List, Literal, Dict, Callable
from datetime import datetime
from abc import ABC, abstractmethod

from .mapping import do_nothing

class DataStrategy(ABC):
    def __init__(
            self,
            dataset_path: str,
            dataset_name: str,
            action_type: Literal["obj", "text"] = "obj"
        ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.action_type = action_type

        # Initialize with default templates and feature functions
        self.templates = self._get_default_templates()
        self.feature_funcs = self._get_default_feature_funcs()

    @abstractmethod
    def _get_default_templates(self) -> Dict[str, str]:
        """Return default templates for the dataset.
        
        Returns:
            Dict mapping template names to template strings
        """
        pass

    @abstractmethod
    def _get_default_feature_funcs(self) -> Dict[str, Callable]:
        """Return default feature functions for the dataset.
        
        Returns:
            Dict mapping feature names to feature extraction functions
        """
        pass

    def set_template(self, template_name: str, template: str) -> None:
        """Set a custom template for a specific feature type."""
        if template_name not in ["action_list", "history", "profile"]:
            raise ValueError(f"Invalid template name: {template_name}")
        self.templates[template_name] = template

    def set_feature_func(self, feature_name: str, func: Callable) -> None:
        """Set a custom feature function for a specific feature type."""
        if feature_name not in ["action_list", "history", "profile"]:
            raise ValueError(f"Invalid feature name: {feature_name}")
        self.feature_funcs[feature_name] = func

