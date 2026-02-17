import torch
import torch.nn.functional as F
from anndata import AnnData
from typing import Dict, List


class ContrastMetrics: 
    """
    Metrics for tracking alignment between two latent variables as part of contrastive training
    """
    def __init__(self): 
        
        super().__init__()
        
        self.agg_metrics = {
            'cosine_sim': F.cosine_similarity, 
                            }

    def __call__(self, z1, z2):
        ret_metrics = {}
        for key, metrics in self.agg_metrics.items(): 
            ret_metrics[key] = metrics(z1, z2).mean().item()
        return ret_metrics


class ReconMetrics: 
    """
    Metrics for gene reconstruction performance
    """
    def __init__(self): 
        super().__init__()
        self.agg_metrics = {
            'mse': F.mse_loss,
            'spearman': batch_spearman,
            'pcc': batch_pearson,
            'pred_equal_var': frac_zero_var,
        }
        
    def __call__(self, preds, target):
        
        ret_metrics = {}
        for key, metric in self.agg_metrics.items(): 
            ret_metrics[key] = metric(preds, target).item()
        return ret_metrics
    
    
    def gene_level_metrics(self, preds: torch.Tensor, target: torch.Tensor): 
        pass


class CalcMetrics: 
    def __init__(self, metrics: List[str] = ['img', 'gene', 'contrast']):         
        super().__init__()
        self.gene_metrics = {
            'gene_mse': F.mse_loss,
            'gene_spearman': batch_spearman,
            'gene_pcc': batch_pearson,
            # 'gene_pcc': 
            }
        self.img_metrics = { 
            'img_mse': F.mse_loss,
            'img_spearman': batch_spearman,
            }
        
        self.contrast_metrics = {
            'cosine_sim': F.cosine_similarity, 
                            }
        
        self.metrics_list = metrics
        
        dict_mapping = {
        'gene': self.gene_metrics,
        'img': self.img_metrics,
        'contrast': self.contrast_metrics,
        }

        self.all_metrics = {
            k: v for m in metrics for k, v in dict_mapping[m].items()
        }
        
        
        # self.all_keys = set(self.recon_metrics.keys()).union(set(self.contrast_metrics.keys()))
        
    def keys(self): 
        return self.all_metrics.keys()
        
            
    def __call__(self, x1, x2, metrics: str, metrics_dict: Dict):
        """
        x1 -> preds | z1
        x2 -> target | z2
        metrics_dict: current dict with keys and AverageMeter
        """        
        assert metrics in self.metrics_list, f"metric must be either {self.metrics_list}, got {metrics}"
        
        if metrics == 'contrast':
            for key, metrics in self.contrast_metrics.items(): 
                metrics_dict[key].update(metrics(x1, x2).mean().item())
            return metrics_dict
                
        elif metrics == 'gene': 
            for key, metrics in self.gene_metrics.items(): 
                metrics_dict[key].update(metrics(x1, x2).mean().item())
            return metrics_dict
        
        elif metrics == 'img': 
            for key, metrics in self.img_metrics.items(): 
                metrics_dict[key].update(metrics(x1, x2).mean().item())
            return metrics_dict
        

def ssim(coords, ad_y: AnnData, ad_x: AnnData) -> float: 
    """
    Given two slide-level anndata objects, compute the structural similarity index
    between the two.
    """



def frac_zero_var(y, x): 
    """
    Check fraction in batch where all predictions are equal.
    """
    return (y.std(dim=1) == 0).float().mean()


def compute_ranks(x):
    # Compute ranks correctly, starting from 1
    sorted_idx = x.argsort(dim=-1)
    ranks = torch.zeros_like(x, dtype=torch.float)
    
    # arange should start at 1 and end at n_genes
    arange = torch.arange(1, x.size(-1)+1, device=x.device, dtype=torch.float).expand_as(x)
    ranks.scatter_(-1, sorted_idx, arange)
    
    return ranks


def batch_spearman(y, x, agg: str = 'mean'):
    y_rank = compute_ranks(y)
    x_rank = compute_ranks(x)

    x_centered = x_rank - x_rank.mean(dim=-1, keepdim=True)
    y_centered = y_rank - y_rank.mean(dim=-1, keepdim=True)

    cov = (x_centered * y_centered).sum(dim=-1) / (y.size(-1) - 1)

    eps = 1e-8
    std_x = x_centered.std(dim=-1, unbiased=True) + eps
    std_y = y_centered.std(dim=-1, unbiased=True) + eps

    spearman_per_sample = cov / (std_x * std_y)

    if agg is None:
        return spearman_per_sample
    else:
        return spearman_per_sample.mean()
    
    
def batch_pearson(y: torch.Tensor, x: torch.Tensor, agg: str = 'mean'):
    """
    Computes the Pearson Correlation Coefficient (PCC) for a batch of predictions and targets.

    Args:
        y (torch.Tensor): The predictions from the model. Shape: (batch_size, num_features).
        x (torch.Tensor): The ground truth targets. Shape: (batch_size, num_features).
        agg (str): The aggregation method. 'mean' returns the mean of the batch, 
                   None returns the PCC for each sample.

    Returns:
        torch.Tensor: The computed PCC, either as a scalar (mean) or a vector.
    """
    # 1. Center the data by subtracting the mean
    # This is the first step for both covariance and standard deviation.
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)

    # 2. Calculate the covariance
    # cov(X, Y) = E[(X - E[X])(Y - E[Y])]
    # For a sample, this is sum((x_i - x_mean) * (y_i - y_mean)) / (n - 1)
    cov = (x_centered * y_centered).sum(dim=-1) / (x.size(-1) - 1)

    # 3. Calculate the standard deviations
    # We add a small epsilon for numerical stability to avoid division by zero.
    # `unbiased=True` makes the denominator n-1, matching the sample covariance.
    eps = 1e-8
    std_x = x_centered.std(dim=-1, unbiased=True) + eps
    std_y = y_centered.std(dim=-1, unbiased=True) + eps

    # 4. Compute the Pearson Correlation Coefficient
    # PCC = cov(X, Y) / (std(X) * std(Y))
    pearson_per_sample = cov / (std_x * std_y)

    # 5. Aggregate the results if required
    if agg is None:
        return pearson_per_sample
    else:
        return pearson_per_sample.mean()