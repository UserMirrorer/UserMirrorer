import os
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime

import random
from typing import List, Callable, Tuple

from safetensors.torch import save_file, load_file
from ..formatter.strategy import DataStrategy


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

def get_text_embedding(texts: pd.Series, llm_embedding) -> List[torch.Tensor]:
    """Convert text descriptions to embeddings using a language model.
    
    Args:
        texts: Series of text descriptions to embed
        llm_embedding: Language model for generating embeddings
        
    Returns:
        List of embedding tensors for each text
    """
    outputs = llm_embedding.embed(texts.apply(json.dumps).tolist())
    return torch.tensor([output.outputs.embedding for output in outputs])

def get_collaborative_embedding(user_item_interaction_matrix: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    """Generate collaborative embeddings from user-item interactions using SVD.
    
    Args:
        user_item_interaction_matrix: Sparse matrix of user-item interactions
        num_eigenvectors: Number of eigenvectors to use
        
    Returns:
        Matrix of collaborative embeddings
    """
    return get_spectral_info(user_item_interaction_matrix, num_eigenvectors=64)

def get_spectral_info(inter_mat: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    """Get spectral information from interaction matrix, including node degrees and singular vectors.
    
    Args:
        inter_mat: Sparse interaction matrix
        num_eigenvectors: Number of eigenvectors to use in SVD
        
    Returns:
        V_mat: Matrix of eigenvectors scaled by degree
    """
    def _degree(mat, dim=0, exp=-0.5):
        """Get degree of nodes"""
        d_inv = torch.nan_to_num(
            torch.clip(torch.sparse.sum(mat, dim=dim).to_dense(), min=1.).pow(exp), 
            nan=1., posinf=1., neginf=1.
        )
        return d_inv
    
    # Calculate degrees and normalized adjacency
    di_isqr = _degree(inter_mat, dim=0).reshape(-1, 1)
    di_sqr = _degree(inter_mat, dim=0, exp=0.5).reshape(-1, 1)

    vals = inter_mat.values() * di_isqr[inter_mat.indices()[1]].squeeze()
    inter_mat = torch.sparse_coo_tensor(
        inter_mat.indices(),
        _degree(inter_mat, dim=1)[inter_mat.indices()[0]] * vals,
        size=inter_mat.shape, dtype=torch.float
    ).coalesce()
    # Get eigenvectors
    _, _, V_mat = torch.svd_lowrank(inter_mat, q=max(4 * num_eigenvectors, 32), niter=10)
    V_mat = V_mat[:, :num_eigenvectors]
    
    return V_mat * di_sqr

def content_predict(
        all_item_embeedding: torch.Tensor,      # (num_items, embedding_dim)
        target_item_id: torch.Tensor           # (batch_size, )
    ) -> torch.Tensor:
    """Generate content-based similarity scores between target items and all items.
    
    Args:
        all_item_embeedding: Embeddings for all items
        target_item_id: IDs of target items to compare against
        
    Returns:
        Similarity scores between target items and all items
    """
    item_id = torch.tensor(target_item_id.tolist()).to(torch.long)
    
    all_scores = torch.matmul(all_item_embeedding[item_id], all_item_embeedding.T)       # (batch_size, num_items)
    return all_scores

def collaborative_predict(
        all_item_embeedding: torch.Tensor,      # (num_items, embedding_dim)
        interaction_matrix: torch.Tensor       # (batch_size, num_items)
    ) -> torch.Tensor:
    """Generate collaborative filtering predictions based on user interactions.
    
    Args:
        all_item_embeedding: Embeddings for all items
        interaction_matrix: Matrix of user-item interactions
        
    Returns:
        Predicted scores for all items based on collaborative filtering
    """
    all_scores = torch.matmul(torch.matmul(interaction_matrix, all_item_embeedding), all_item_embeedding.T)        # (batch_size, num_items)
    return all_scores


class SampleFormulator(object):
    """Class for generating training samples by combining different recommendation approaches.
    
    This class handles:
    1. Loading and preprocessing item, user and interaction data
    2. Computing content and collaborative embeddings
    3. Combining multiple scoring methods
    4. Sampling items based on scores
    5. Filtering sampled items
    6. Adding ground truth items
    """
    def __init__(
            self,
            ds: DataStrategy,
            item_num_range: Tuple[int, int] = (16, 32),
            llm_embedding = None
        ):
        self.ds = ds
        self.item_num_range = item_num_range
        self.scoring_methods = []
        self.sampling_method = None
        self.filtering_methods = []

        self.set_sampling_method("topk")
        for method in ["random"]:
            self.add_scoring_method(method)

        self._load_data()
        self._construct_id_mapping()
        self._construct_interaction_matrix()


    def _load_data(self):
        """
        Load user, item, and interaction data from JSON files.
        
        Reads three types of data:
        - User features from *_user_feature.jsonl
        - Item features from *_item_feature.jsonl
        - Interaction data from *_interaction.jsonl
        
        The interaction data is processed to include chronological ordering.
        """
        self.user_df = pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_user_feature.jsonl"), lines=True)
        self.item_df = pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_item_feature.jsonl"), lines=True)
        self.interaction_df = get_interaction_order(pd.read_json(os.path.join(self.ds.dataset_path, "raws", f"{self.ds.dataset_name}_interaction.jsonl"), lines=True))
        if os.path.exists(os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.collaborative.safetensors")):
            embeddings = load_file(os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.collaborative.safetensors"))
            self.item_collaborative_embedding = embeddings["item_collaborative_embedding"]
        else:
            self.item_collaborative_embedding = None
        if os.path.exists(os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.content.safetensors")):
            embeddings = load_file(os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.content.safetensors"))
            self.item_content_embedding = embeddings["item_content_embedding"]
        else:
            self.item_content_embedding = None

    def _construct_interaction_matrix(self):
        """Construct sparse interaction matrix from user-item interactions.
        Creates a matrix where rows are users, columns are items, and values indicate interactions.
        Also builds a dictionary mapping users to their interaction histories.
        """
        self.user_item_interaction_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(self.interaction_df[['uid', 'iid']].values).T,
            values=torch.ones(self.interaction_df.shape[0]),
            size=(self.interaction_df['uid'].nunique(), self.interaction_df['iid'].nunique())
        ).coalesce()
        self.user_interaction_list = self.interaction_df.groupby('user_id')['item_id'].agg(list)

    def _construct_id_mapping(self):
        """Create mappings between original IDs and internal integer indices.
        Maps user_id/item_id to uid/iid for efficient matrix operations.
        """
        self.item_id_map = {item_id: i for i, item_id in enumerate(self.interaction_df['item_id'].unique())}
        self.user_id_map = {user_id: i for i, user_id in enumerate(self.interaction_df['user_id'].unique())}
        self.interaction_df['iid'] = self.interaction_df['item_id'].map(self.item_id_map)
        self.interaction_df['uid'] = self.interaction_df['user_id'].map(self.user_id_map)
        self.item_df['iid'] = self.item_df['item_id'].map(self.item_id_map)
        self.user_df['uid'] = self.user_df['user_id'].map(self.user_id_map)
        self.item_df = self.item_df.set_index('iid')
        self.user_df = self.user_df.set_index('uid')

    def construct_collaborative_features(self):
        """Generate item collaborative embeddings if not provided.
        
        Args:
            llm_embedding: Language model for generating text embeddings
        """
        if self.item_collaborative_embedding is None:
            print("Generating item collaborative embedding")
            self.item_collaborative_embedding = get_collaborative_embedding(self.user_item_interaction_matrix, num_eigenvectors=128)

        if not os.path.exists(os.path.join(self.ds.dataset_path, "embeddings")):
            os.makedirs(os.path.join(self.ds.dataset_path, "embeddings"))
        save_file(
            {
                "item_collaborative_embedding": self.item_collaborative_embedding
            },
            filename=os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.collaborative.safetensors")
        )

    def construct_content_features(self, llm_embedding):
        """Generate content embeddings if not provided.
        Args:
            llm_embedding: Language model for generating text embeddings
        """
        if self.item_content_embedding is None:
            print("Generating content embedding")
            self.item_content_embedding = get_text_embedding(self.item_df['item_description'], llm_embedding)

        if not os.path.exists(os.path.join(self.ds.dataset_path, "embeddings")):
            os.makedirs(os.path.join(self.ds.dataset_path, "embeddings"))
        save_file(
            {
                "item_content_embedding": self.item_content_embedding,
            },
            filename=os.path.join(self.ds.dataset_path, "embeddings", f"{self.ds.dataset_name}.content.safetensors")
        )

    def add_scoring_method(self, method: Callable | str, interaction_index: bool = False):
        if isinstance(method, Callable):
            self.scoring_methods.append((method, interaction_index))
        elif isinstance(method, str):
            if method == "popularity":
                self.scoring_methods.append((popularity_scoring, True))
            elif method == "content-based":
                self.scoring_methods.append((content_based_scoring, False))
            elif method == "collaborative":
                self.scoring_methods.append((collaborative_scoring, True))
            elif method == "random":
                self.scoring_methods.append((random_scoring, False))
            else:
                raise ValueError(f"Invalid scoring method: {method}")
            

    def add_filtering_method(self, method: Callable | str):
        if isinstance(method, Callable):
            self.filtering_methods.append(method)
        elif isinstance(method, str):
            if method == "interaction":
                self.filtering_methods.append(interaction_filtering)
            else:
                raise ValueError(f"Invalid filtering method: {method}")

    def set_sampling_method(self, method: Callable | str):
        if isinstance(method, Callable):
            self.sampling_method = method
        elif isinstance(method, str):
            if method == "topk":
                self.sampling_method = topk_sampling
            elif method == "multinomial":
                self.sampling_method = multinomial_sampling
        else:
            raise ValueError(f"Invalid sampling method: {method}")


    def sampling(self, n_samples: int = 16, min_interaction_cnt: int = 3, user_ids: List[int] = None) -> torch.Tensor:
        """Generate training samples by combining multiple recommendation approaches.
        
        Process:
        1. Sample random interactions
        2. Score items using multiple methods
        3. Sample items based on scores
        4. Merge ranked lists
        5. Apply filtering
        6. Add ground truth items
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame containing sampled interactions with impression lists
        """
        # Random sampling from interaction pool
        cand_inter_df = self.interaction_df[self.interaction_df['order_id'] >= min_interaction_cnt]
        if user_ids is not None:
            cand_inter_df = cand_inter_df[cand_inter_df['user_id'].isin(pd.Series(user_ids))]
        sampled_inter_ids = cand_inter_df.sample(n=min(n_samples, cand_inter_df.shape[0])).index

        all_sampled_item_ids = []
        for scoring_method, interaction_index in self.scoring_methods:
            scores = torch.softmax(scoring_method(self, sampled_inter_ids), dim=1)
            
            sampled_item_ids = self.sampling_method(scores, sample_num=max(32, self.item_num_range[1] * 3))
            sampled_item_ids = sampled_item_ids.numpy().tolist()

            # Map iid to item_id
            if interaction_index:
                sampled_item_ids = [[self.item_df['item_id'].loc[i] for i in item_list] for item_list in sampled_item_ids]
            else:
                sampled_item_ids = [[self.item_df['item_id'].iloc[i] for i in item_list] for item_list in sampled_item_ids]
            
            all_sampled_item_ids.append(sampled_item_ids)
        all_sampled_item_ids = pd.DataFrame(all_sampled_item_ids).T

        all_sampled_item_ids = all_sampled_item_ids.progress_apply(list, axis=1).apply(merge_ranking_list)
        sampled_inter = self.interaction_df.loc[sampled_inter_ids].copy().reset_index(drop=True)
        sampled_inter['impression_list'] = all_sampled_item_ids

        # Filtering
        if self.filtering_methods is not None:
            for filtering_method in self.filtering_methods:
                sampled_inter = filtering_method(self, sampled_inter)

        # Clip the number of items
        samp_range = self.item_num_range[0] - 1, self.item_num_range[1] - 1
        sampled_inter['impression_list'] = sampled_inter['impression_list'].apply(lambda x: x[:random.randint(samp_range[0], samp_range[1])])
        impression_cnt = sampled_inter['impression_list'].apply(len)
        sampled_inter = sampled_inter[impression_cnt >= samp_range[0]]

        # Add Ground Truth
        if 'item_pos' not in sampled_inter.columns:
            sampled_inter['impression_list'], sampled_inter['item_pos'] = zip(*sampled_inter.apply(lambda x: add_gt_to_impression(x['impression_list'], x['item_id']), axis=1))
        
        return sampled_inter

# Scoring Methods
def random_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Generate random scores for items.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Random scores for each item
    """
    return torch.rand(sampled_inter_ids.shape[0], sf.item_df.shape[0])

def popularity_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on their popularity (interaction count).
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Popularity scores for each item
    """
    pop_scores = torch.tensor(sf.interaction_df['iid'].value_counts().sort_index().values).to(torch.float).repeat(sampled_inter_ids.shape[0], 1)
    return pop_scores

def content_based_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on content similarity using item embeddings.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Content-based similarity scores for each item
    """
    return content_predict(sf.item_content_embedding, sf.interaction_df['iid'].loc[sampled_inter_ids])

def collaborative_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on collaborative filtering using interaction patterns.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Collaborative filtering scores for each item
    """
    user_ids = sf.interaction_df['uid'].loc[sampled_inter_ids].to_numpy()
    inter_mat = sf.user_item_interaction_matrix.index_select(0, torch.tensor(user_ids).to(torch.long)).to_dense()
    return collaborative_predict(sf.item_collaborative_embedding, inter_mat)

# Sampling Methods
def multinomial_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    """Sample items using multinomial probability distribution.
    
    Args:
        scores: Probability scores for each item
        sample_num: Number of items to sample
        
    Returns:
        Indices of sampled items
    """
    assert sample_num <= scores.shape[1], "Sample number is larger than the number of items"
    return torch.multinomial(scores, num_samples=sample_num)

def topk_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    """Sample top-k items with highest scores.
    
    Args:
        scores: Scores for each item
        sample_num: Number of top items to select
        
    Returns:
        Indices of top-k scored items
    """
    return torch.topk(scores, k=sample_num).indices

# Filtering Methods
def interaction_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items that user has already interacted with.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with previously interacted items removed
    """
    # sampled_inter['interaction_list'] = sf.user_interaction_list.loc[sampled_inter['user_id']]
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if i not in sf.user_interaction_list.loc[x['user_id']]],
        axis=1
    )
    return sampled_inter

def time_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items based on timestamp constraints.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with future items removed
    """
    time = sf.item_df.set_index('item_id')['item_features'].apply(lambda x: datetime.fromtimestamp(x['time']))
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if time[i] <= x['timestamp']],
        axis=1
    )
    return sampled_inter

# Inserting Ground Truth
def gt_position(ranked_list: List) -> int:
    """Randomly determine position to insert ground truth item.
    
    Args:
        ranked_list: List of ranked items
        
    Returns:
        Random position index for ground truth insertion
    """
    pos = random.randint(0, len(ranked_list) - 1)
    return pos

def add_gt_to_impression(impression_list: List, item_id: str) -> Tuple[List, int]:
    """Insert ground truth item into impression list at random position.
    
    Args:
        impression_list: List of item impressions
        item_id: Ground truth item ID to insert
        
    Returns:
        Updated impression list with ground truth item inserted
    """
    pos = gt_position(impression_list)
    impression_list.insert(pos, item_id)
    return impression_list, pos

# Merge Ranking List Randomly
def merge_ranking_list(ranking_lists: List[List]) -> List:
    """Merge multiple ranking lists while preserving relative ordering within each list.
    
    Args:
        ranking_lists: List of ranked item lists to merge
        
    Returns:
        Single merged list with duplicates removed
    """
    ranking_lists = [lst for lst in ranking_lists if len(lst) > 0]
    if not ranking_lists:
        return []
    length_list = np.array([len(ranking_list) for ranking_list in ranking_lists]).cumsum()
    idx_list = np.arange(length_list[-1])
    random.shuffle(idx_list)
    # Sort the idx_list in each section, split by the length of the ranking_lists
    split_idx_list = np.array_split(idx_list, length_list)
    merged_list = np.empty(length_list[-1], dtype=object)
    for ranking_list, split_idx in zip(ranking_lists, split_idx_list):
        merged_list[np.sort(split_idx)] = np.array(ranking_list)
    return list(dict.fromkeys(merged_list))     # Remove duplicates
