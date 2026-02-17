"""
Evaluation class to be plugged in each iteration of training
"""
from __future__ import annotations
import numpy as np
import torch
from pprint import pprint
from anndata import AnnData
from itertools import chain
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

try: 
    from cuml.linear_model import Ridge
except ImportError:
    from sklearn.linear_model import Ridge
from typing import Dict
from tqdm import tqdm
import tifffile as tifffile


def run_linprobe(Z_train: np.ndarray, 
                 Z_test: np.ndarray, 
                 y_train: np.ndarray, 
                 y_test: np.ndarray, 
                 test_genes:Dict, 
                 test_gene_index: list,
                 model: nn.Module, 
                 pca_reduce:bool=True, 
                 random_state:int=42, 
                 folds: int = 5,
                 use_decoder: bool = False,
                 ): 
    
    all_test_genes = list(set(chain.from_iterable(test_genes.values())))
    all_raw_results = {}
    all_results = {}
    mean_results = {}
    std_results = {}
    fold_results = {}
    
    
    for fold in range(folds): 
        print(f"Fold {fold+1}...")
        seed = random_state + fold
    
        Z_train_fold, Z_test_fold, y_train_fold, y_test_fold = train_test_split(
            Z_train,
            y_train,
            train_size=0.8,
            random_state=fold
        )
        
        # Z_train, Z_test, y_train, y_test = train_test_split(
        # # X.numpy() if isinstance(X, torch.Tensor) else X,
        # # y.numpy() if isinstance(y, torch.Tensor) else y,
        #     test_size=0.2,
        #     random_state=random_state
        # )
        if isinstance(Z_train_fold, torch.Tensor): # convert back to tensor if we're using the decoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Z_train, Z_test = torch.from_numpy(Z_train).to(device), torch.from_numpy(Z_test).to(device)
            y_test, y_train = torch.from_numpy(y_test).to(device), torch.from_numpy(y_train).to(device)



        if pca_reduce and not use_decoder: 
            from cuml.preprocessing import StandardScaler
            from cuml import PCA
            from sklearn.pipeline import Pipeline
            print('Perform PCA dim reduction...')
            pipe = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=256, random_state=seed, svd_solver='auto'))])
            Z_train_fold, Z_test_fold = pipe.fit_transform(Z_train_fold), pipe.transform(Z_test_fold)
            
    
        # single fold regression
        if use_decoder: 
            results, dump = train_test_reconstruction(Z_train_fold, Z_test_fold, y_train_fold, y_test_fold, all_test_genes, test_gene_index, model)
        else: 
            results, dump = train_test_reg(Z_train_fold, Z_test_fold, y_train_fold, y_test_fold, genes=all_test_genes, random_state=seed)
        
        all_raw_results[fold] = results
        
    # calculate mean across folds
    for metric in results.keys():
        mean_results[metric] = {}
        std_results[metric] = {}
        fold_results[metric] = {}
        for genes in results[metric].keys(): 
            values = [all_raw_results[fold][metric][genes] for fold in range(folds)]
            fold_results[metric][genes] = values # log list
            mean_results[metric][genes] = np.round(np.mean(values), 8)
            std_results[metric][genes] = np.round(np.std(values), 8)
    
        
    # # reorder results by organ
    for organ, genes in test_genes.items(): 
        organ_results = {}
        organ_results['pearson_corrs_mean'] = {g: mean_results['pearson_corrs'][g] for g in genes}
        organ_results['pearson_corrs_std'] = {g: std_results['pearson_corrs'][g] for g in genes}
        organ_results['pearson_mean'] = np.round(np.mean(list(organ_results['pearson_corrs_mean'].values())), 8)
        organ_results['pearson_std'] = np.round(np.mean(list(organ_results['pearson_corrs_std'].values())), 8)
        
        
        # spearman corrs
        organ_results['spearman_corrs_mean'] = {g: mean_results['spearman_corrs'][g] for g in genes}
        organ_results['spearman_corrs_std'] = {g: std_results['spearman_corrs'][g] for g in genes}
        organ_results['spearman_mean'] = np.round(np.mean(list(organ_results['spearman_corrs_mean'].values())), 8)
        organ_results['spearman_std'] = np.round(np.mean(list(organ_results['spearman_corrs_std'].values())), 8)
        
        # all fold values
        organ_results['pearson_corrs_gene'] = {g: fold_results['pearson_corrs'][g] for g in genes}
        fold_values_by_gene = [fold_results['pearson_corrs'][g] for g in genes]
        organ_results['pearson_corrs'] = [np.mean(fold_scores) for fold_scores in zip(*fold_values_by_gene)]
        
        organ_results['l2_errors'] = {g: mean_results['l2_errors'][g] for g in genes}
        organ_results['l2_errors_std'] = {g: std_results['l2_errors'][g] for g in genes}
        organ_results['l2_mean'] = np.round(np.mean(list(organ_results['l2_errors'].values())), 8)
        organ_results['l2_std'] = np.round(np.mean(list(organ_results['l2_errors_std'].values())), 8)
        
        organ_results['r2_scores'] = {g: mean_results['r2_scores'][g] for g in genes}
        organ_results['r2_scores_std'] = {g: std_results['r2_scores'][g] for g in genes}
        organ_results['r2_mean'] = np.round(np.mean(list(organ_results['r2_scores'].values())), 8)
        organ_results['r2_std'] = np.round(np.mean(list(organ_results['r2_scores_std'].values())), 8)
        
        
        all_results[organ] = organ_results
      
    
    avg_pearson = np.mean([result['pearson_mean'] for result in all_results.values()])
    std_pearson = np.mean([result['pearson_std'] for result in all_results.values()])
    
    
    
    pprint(all_results)
    print(f"Probing Pearson mean: {np.round(avg_pearson, 6)} ± {np.round(std_pearson, 6)}")
    
    return all_results

def train_test_reconstruction(Z_train: np.ndarray, 
                              Z_test: np.ndarray, 
                              y_train: np.ndarray, 
                              y_test: np.ndarray,
                              all_test_genes: list, 
                              test_gene_index: list,
                              model: nn.Module): 
    
    preds_all = []
    errors = {}
    r2_scores = {}
    pearson_corrs = {}
    
    
    # check device
    # X_test = torch.tensor(X_test, device="cuda" if torch.cuda.is_available() else "cpu")
    pred = model.decoder(Z_test)
    pred_filtered = pred[:, test_gene_index].detach().cpu().numpy()
    
    for target_idx in range(pred_filtered.shape[1]):
        preds = pred_filtered[:, target_idx]
        preds_all.append(preds)
            
        target_vals = y_test[:, target_idx].detach().cpu().numpy()
        l2_error = float(np.mean((preds - target_vals)**2))
        from sklearn.metrics import r2_score as sk_r2_score
        r2_score = float(sk_r2_score(target_vals, preds))
        pearson_corr, _ = pearsonr(target_vals, preds)
        spearman_corr, _ = spearmanr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(target_vals)
            print(preds)
        
        pearson_corrs[all_test_genes[target_idx]] = np.round(pearson_corr, 5)
        errors[all_test_genes[target_idx]] = np.round(l2_error, 5)
        r2_scores[all_test_genes[target_idx]] = np.round(r2_score, 5)
    
    results = {'l2_errors': errors, 
            'r2_scores': r2_scores,
            'pearson_corrs': pearson_corrs,
        }
    dump = {
        'preds_all': preds_all,
        'targets_all': y_test,
    }
    
    return results, dump
    



def train_test_reg(X_train, X_test, y_train, y_test,
                   max_iter=1000, random_state=0, 
                   genes=None, 
                   method='ridge'):
    """
    Linear probing of gene expression as regression task
    """

    
    preds_all = []
    errors = {}
    r2_scores = {}
    pearson_corrs = {}
    spearman_corrs = {}
    # pearson_genes = []
    print("Linear probing...")
    for target_idx in tqdm(range(y_test.shape[1])):
        if method == 'ridge':
            alpha = 100 / (X_train.shape[1] * y_train.shape[1])            
            reg = Ridge(solver='svd',
                        fit_intercept=False, 
                        normalize=False,
                        alpha=alpha,
                        )
            
            reg.fit(X_train, y_train[:, target_idx])
            
            preds = reg.predict(X_test)
            preds_all.append(preds)
                
        target_vals = y_test[:, target_idx]
        l2_error = float(np.mean((preds - target_vals)**2))
        # compute r2 score
        from sklearn.metrics import r2_score as sk_r2_score
        r2_score = float(sk_r2_score(target_vals, preds))
        # r2_score = float(1 - np.sum((target_vals - preds)**2) / np.sum((target_vals - np.mean(target_vals))**2))
        pearson_corr, _ = pearsonr(target_vals, preds)
        spearman_corr, _ = spearmanr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(target_vals)
            print(preds)
        
        pearson_corrs[genes[target_idx]] = np.round(pearson_corr, 5)
        errors[genes[target_idx]] = np.round(l2_error, 5)
        r2_scores[genes[target_idx]] = np.round(r2_score, 5)
        spearman_corrs[genes[target_idx]] = np.round(spearman_corr, 5)

    results = {'l2_errors': errors, 
               'r2_scores': r2_scores,
               'pearson_corrs': pearson_corrs,
               'spearman_corrs': spearman_corrs,
            }
    dump = {
        'preds_all': preds_all,
        'targets_all': y_test,
    }
    
    return results, dump


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        alpha (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix

    if embedding_matrix.is_cuda:
        embedding_matrix = embedding_matrix.cpu()

    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank


class SlidePredTracker: 
    """
    Class to track and reconstruct slide-level predictions
    """
    
    def __init__(self):
        super().__init__() 
        self.preds = {}
        # self.gt = {}
        self.adata = []
        
        

    def add(self, preds): 
        """
        Add the predictions and ground truth to the tracker for given metadata
        """
        pass


    def gt_adata(self) -> Dict[str, AnnData]: 
        """
        Construct ground truth and predictions to adata objects, accessible by key
        """
        pass
        
    def pred_adata(self): 
        pass
        
    def gt_img(self): 
        pass
        
    def pred_img(self): 
        pass
