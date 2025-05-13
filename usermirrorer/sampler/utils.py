import torch
import pandas as pd

from typing import List


from .sampler import SampleFormulator

def get_text_embedding(texts: List[str], llm_embedding) -> List[torch.Tensor]:
    outputs = llm_embedding.encode(texts)
    return torch.tensor([output.outputs.embedding for output in outputs])

def get_collaborative_embedding(user_item_interaction_matrix: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    # Perform SVD on the interaction matrix
    return get_spectral_info(user_item_interaction_matrix, num_eigenvectors=64)

# Scoring Methods
def random_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    return torch.rand(sampled_inter_ids.shape[0], sf.item_pool.shape[0])

def popularity_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    pop_scores = torch.tensor(sf.interaction_pool['iid'].value_counts().sort_index().values).to(torch.float).repeat(sampled_inter_ids.shape[0], 1)
    return pop_scores

def content_based_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    return content_predict(sf.item_content_embedding, sf.interaction_pool['iid'].loc[sampled_inter_ids])

def collaborative_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    user_ids = sf.interaction_pool['uid'].loc[sampled_inter_ids].to_numpy()
    inter_mat = sf.user_item_interaction_matrix.index_select(0, torch.tensor(user_ids).to(torch.long)).to_dense()
    return collaborative_predict(sf.item_collaborative_embedding, inter_mat)

def get_spectral_info(inter_mat: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    """Get spectral information from interaction matrix, including node degrees and singular vectors.
    Reference: 
        Shen, Yifei, et al. "How powerful is graph convolution for recommendation?." Proc. CIKM 2021.
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
    all_scores = torch.matmul(all_item_embeedding[target_item_id], all_item_embeedding.T)       # (batch_size, num_items)
    return all_scores

def collaborative_predict(
        all_item_embeedding: torch.Tensor,      # (num_items, embedding_dim)
        interaction_matrix: torch.Tensor       # (batch_size, num_items)
    ) -> torch.Tensor:
    all_scores = torch.matmul(torch.matmul(interaction_matrix, all_item_embeedding), all_item_embeedding.T)        # (batch_size, num_items)
    return all_scores

# Sampling Methods
def multinomial_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    assert sample_num <= scores.shape[1], "Sample number is larger than the number of items"
    return torch.multinomial(scores, num_samples=sample_num)

def topk_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    return torch.topk(scores, k=sample_num)

# Filtering Methods
def interaction_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    sampled_inter['interaction_list'] = sf.user_interaction_list[sampled_inter['uid']]
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if i not in x['interaction_list']],
        axis=1
    )
    return sampled_inter