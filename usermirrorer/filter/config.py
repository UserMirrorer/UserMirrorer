from typing import List
from dataclasses import dataclass, field


@dataclass
class FilteringConfig:
    name: str = field(default="UserMirrorer")
    mode: str = field(default="decision")
    project_path: str = field(default="/")
    filtered_num: int = field(default=10000)
    filter_column: str = field(default="diff_model_uncertainty")
    sampling_column: str = field(default="entropy")
    ascending: bool = field(default=False)
    group_columns: List[str] = field(default_factory=lambda: ["dataset", "choice_cnt"])
    datasets: List[str] = field(default_factory=lambda: ["KuaiRec2", "LastFM", "MIND",
        "mobilerec", "Amazon-Beauty",
        "Amazon-Fashion", "Amazon-Office",
        "Amazon-Grocery", "ml-1m", "steam", "goodreads"
    ])
