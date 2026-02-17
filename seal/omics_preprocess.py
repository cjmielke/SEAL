from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scanpy as sc
import yaml
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def preprocess_spot(sample, config_path: str = "conf/config.yaml") -> Tuple[AnnData, np.ndarray]:
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    
    n_train_genes = int(cfg["n_train_genes"])
    assert n_train_genes > 0, "n_train_genes must be a positive integer"
    

    gene_norm = cfg.get("gene_norm", "log_norm")
    gene_criterion = cfg.get("gene_criterion", "hvg")
    spot_context = bool(cfg.get("spot_context", True))
    st_technology = sample.meta.get("st_technology")

    adata = sample.adata.copy()
    adata.var_names = adata.var_names.str.upper()
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    adata = adata[
        :,
        ~adata.var_names.str.contains("BLANK|CONTROL|AMBIGUOUS", case=False, regex=True),
    ].copy()

    x = adata.X.toarray().astype(np.float32) if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float32)
    coords = adata.obs[["array_row", "array_col"]].to_numpy()

    if st_technology in ["Visium", "Visium HD"] and len(coords) > 1 and np.isfinite(coords).all():
        g = NearestNeighbors(radius=np.sqrt(2), algorithm="kd_tree").fit(coords).radius_neighbors_graph(coords, mode="connectivity")
        cnt = np.asarray(g.sum(axis=1)).ravel()
        cnt[cnt == 0] = 1
        x_smooth = 0.5 * (g.dot(x) / cnt[:, None]) + 0.5 * x
    else:
        x_smooth = x

    adata.layers["X_smooth"] = csr_matrix(x_smooth.astype(np.float32))

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)

    if gene_norm == "log_norm":
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.normalize_total(adata, target_sum=1e4, layer="X_smooth", inplace=True)
        sc.pp.log1p(adata, layer="X_smooth")

    flavor = "seurat" if gene_norm in ["log", "log_norm"] else "seurat_v3"

    if gene_criterion == "heg":
        adata.var["mean_expression"] = np.ravel(adata.X.mean(0))
        genes = adata.var.sort_values("mean_expression", ascending=False).index[:n_train_genes]
    else:
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_train_genes, inplace=True)
        sort_col = "variances_norm" if flavor == "seurat_v3" else "dispersions_norm"
        genes = adata.var.sort_values(sort_col, ascending=False).index[:n_train_genes]

    adata = adata[:, genes].copy()

    x_gene = adata.layers["X_smooth"] if spot_context else adata.X
    x_gene = x_gene.toarray() if hasattr(x_gene, "toarray") else np.asarray(x_gene)
    x_gene = x_gene.astype(np.float32)

    return adata, x_gene
