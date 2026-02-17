from __future__ import annotations

import torch.nn as nn
from anndata import AnnData
from operator import itemgetter
from tqdm import tqdm
from accelerate import Accelerator
from itertools import chain
import torch.nn.functional as F
from typing import *
from seal.models.components import create_mlp
import os
import torch
from torch.cuda.amp import autocast 
from seal.losses.gene_loss import get_gene_loss_fn
from seal.utils.metrics import CalcMetrics
from seal.utils.eval_utils import run_linprobe, AverageMeter, SlidePredTracker
try:
    from cuml import PCA  # type: ignore[import-not-found]
except Exception:
    from sklearn.decomposition import PCA



class GeneMLP(nn.Module):
    def __init__(self, 
                 in_dim=512,
                 out_dim=256, 
                 hidden_dims: List[int] = [512],
                 end_with_fc: bool=True, 
                 mask_ratio: float = None, 
                 ):
        super().__init__()
        
        self.mask_ratio = mask_ratio        
        
        self.projection = create_mlp(in_dim=in_dim, 
                                          hid_dims=hidden_dims,
                                          dropout=0.25,
                                          act=nn.GELU(),
                                          out_dim=out_dim,
                                          batch_norm=True,
                                          end_with_fc=end_with_fc)
        
        self.decoder = get_gene_decoder(decoder='mlp', 
                                        embed_dim=out_dim, 
                                        out_dim=in_dim
                                        )
        

    def forward(self, gene):
        if self.mask_ratio is not None: 
            mask = (torch.rand(gene.shape) > self.mask_ratio).float().to(gene.device)
            if self.training: 
                gene = gene * mask
            else: 
                pass
        
        gene = gene.to(self.projection[0].weight.dtype)
        z = self.projection(gene)
        
        y = self.decoder(z)
        
        return {
            'z': z, 
            'y': y, 
        }
        
        
class ModelDummy(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.decoder = self.decode
    def forward(self, x): 
        return x + (0.0 * self.dummy_param)
    def decode(self, x): 
        return x
        


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # If dimensions differ, define a learnable skip to match shapes
        self.skip_proj = None
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        if self.skip_proj is not None:
            identity = self.skip_proj(x)
        out = out + identity
        out = self.relu(out)
        return out
    
class PlanarFlow(nn.Module):
    """Planar normalizing flow layer: z -> z' = z + u * tanh(w^T z + b)."""
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, z):
        """
        z: Tensor of shape (N, dim) representing latent samples.
        Returns: z_transformed, log_abs_det_jacobian (per sample).
        """
        w_dot_z = torch.sum(z * self.w.unsqueeze(0), dim=1, keepdim=True) + self.b  # (N, 1)
        
        # tanh activation
        activation = torch.tanh(w_dot_z)  # (N, 1)
        
        # Transform: z' = z + u * tanh(w^T z + b)
        z_transformed = z + self.u.unsqueeze(0) * activation  # (N, dim)
        
        phi_prime = 1 - activation**2  # (N, 1)
        u_dot_w = torch.sum(self.u * self.w)  # scalar
        
        # log |det(J)| = log |1 + ψ^T u| where ψ = φ'(w^T z + b) * w
        log_abs_det = torch.log(torch.abs(1 + phi_prime.squeeze(1) * u_dot_w) + 1e-8)  # (N,)
        
        return z_transformed, log_abs_det


class GeneVAE(nn.Module):
    def __init__(self, 
                 in_dim=2000, 
                 latent_dim=512, 
                 hidden_dims=[1024, 512], 
                 dropout: float = 0.0,
                 dec_dropout: float = 0.4,
                 n_flows=3,
                 mask_ratio: float = 0.0,
                 projection_head: str = None,
                 dec_batch_norm: bool = False, 
                 ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        # Encoder
        encoder_layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hd))
            encoder_layers.append(nn.BatchNorm1d(hd))
            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hd
        self.encoder = nn.Sequential(*encoder_layers)
        
        if projection_head == 'linear': 
            self.projection_head = nn.Linear(latent_dim, latent_dim)
        elif projection_head == 'mlp':
            self.projection_head = create_mlp(in_dim=latent_dim, 
                                                out_dim=latent_dim, 
                                                batch_norm=False)
        else: 
            self.projection_head = None
        
        # add normalizing flows
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
        
        # Latent mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hd in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hd))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = hd
            if dec_batch_norm: 
                decoder_layers.append(nn.BatchNorm1d(hd))
        decoder_layers.append(nn.Dropout(dec_dropout))
        
        decoder_layers.append(nn.Linear(prev_dim, in_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def forward(self, x, projection: bool = False):
        
        # Optional Masking
        if self.mask_ratio > 0.0 and self.training: 
            # create random 0-mask for `mask_ratio` fraction of genes
            mask = (torch.rand_like(x) < self.mask_ratio)
            # x = x * mask
            x = x.masked_fill(mask, 0.0) # set masked genes to 0
            
            assert (x[mask].sum() == 0.0), "Masked genes should be set to 0"
        else: 
            mask = torch.zeros_like(x).bool() # during inference, use all genes    
    
    
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        total_log_det = torch.zeros(z.size(0), device=z.device)  # Initialize per batch
        
        if len(self.flows) > 0: 
            for flow in self.flows: 
                z, log_det = flow(z)
                total_log_det += log_det
    
        
        y = self.decoder(z)
        
        if self.projection_head is not None: 
            z_proj = self.projection_head(z)
        if projection: 
            z = z_proj if self.projection_head is not None else z
        
        return {
            'z': z, 
            'y': y,
            'mu': mu,
            'logvar': logvar,
            'total_log_det': total_log_det,
            'mask': mask,
        }
    
    def compute_loss(self, outputs: Dict, x_gene: torch.tensor, beta: float = 0.0):

        logvar = outputs['logvar']
        mu = outputs['mu']
        total_log_det = outputs['total_log_det']
        y_gene = outputs['y'] # reconstructed genes
        # x_gene = outputs['x'] # original input genes
        mask = outputs['mask']
        
        scale_factor = 10
        if mask.any(): 
            assert (x_gene[mask].sum() > 0.0)
            # masked reconstruction loss
            mlm = F.mse_loss(y_gene[mask], x_gene[mask], reduction='mean')
        else: 
            mlm = torch.tensor(0.)
            
        
                
        # base KL: KL[q_0(z_0) || N(0,I)]
        base_kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)  # (batch_size,)
        
        # With flows: KL[q_K(z_K) || N(0,I)] = KL[q_0(z_0) || N(0,I)] - sum(log|det(J_i)|)
        kl_divergence = torch.mean(base_kl - total_log_det)  # Average over batch
        
        # model_loss = (beta * kl_divergence) + mlm
        
        loss_dict = {
            'kl_loss': beta * kl_divergence, 
            'mlm': mlm * scale_factor,
        }
        
        return loss_dict
    
    
    
# class GeneVAE(nn.Module):
#     def __init__(self, 
#                  in_dim=2000, 
#                  latent_dim=512, 
#                  hidden_dims=[1024, 512], 
#                  n_flows=3,
#                  ):
#         super().__init__()
        
#         # Encoder
#         encoder_layers = []
#         prev_dim = in_dim
#         for hd in hidden_dims:
#             encoder_layers.append(nn.Linear(prev_dim, hd))
#             encoder_layers.append(nn.BatchNorm1d(hd))
#             encoder_layers.append(nn.ReLU(inplace=True))
#             prev_dim = hd
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # add normalizing flows
#         self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
        
#         # Latent mean and log-variance
#         self.fc_mu = nn.Linear(prev_dim, latent_dim)
#         self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
#         # Decoder
#         decoder_layers = []
#         prev_dim = latent_dim
#         for hd in reversed(hidden_dims):
#             decoder_layers.append(nn.Linear(prev_dim, hd))
#             decoder_layers.append(nn.ReLU(inplace=True))
#             prev_dim = hd
#         decoder_layers.append(nn.Linear(prev_dim, in_dim))
#         self.decoder = nn.Sequential(*decoder_layers)
        
#         # add cosine annealing schedule for KL penalty
#         self.warmup_epochs = 5

#     def reparameterize(self, mu, logvar):
#         # Reparameterization trick
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         # Encode
#         h = self.encoder(x)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
        
#         # Reparameterize
#         z = self.reparameterize(mu, logvar)
        
#         # pass through flows
#         total_log_det = 0.0
#         if len(self.flows) > 0: 
#             for flow in self.flows: 
#                 z, log_det = flow(z)
#                 total_log_det += log_det
        
#         y = self.decoder(z)
        
        
#         # if decode: 
#         return {
#             'z': z, 
#             'y': y,
#             'mu': mu,
#             'logvar': logvar,
#             'total_log_det': total_log_det
#         }
    
#     def compute_loss(self, outputs: Dict, beta: float = 1.0):
#         """
#         Calculate kl_penalty after the foward pass
#         """
#         logvar = outputs['logvar']
#         mu = outputs['mu']
        
#         pen = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
        
#         kl_penalty = beta * pen
        
#         return kl_penalty
    
    
 
# class PlanarFlow(nn.Module):
#     """Planar normalizing flow layer: z -> z' = z + u * tanh(w^T z + b)."""
#     def __init__(self, dim):
#         super(PlanarFlow, self).__init__()
#         # Initialize u, w as small random vectors and b as zero
#         self.u = nn.Parameter(torch.randn(dim) * 0.01)
#         self.w = nn.Parameter(torch.randn(dim) * 0.01)
#         self.b = nn.Parameter(torch.zeros(1))
    
#     def forward(self, z):
#         """
#         z: Tensor of shape (N, dim) representing latent samples.
#         Returns: z_transformed, log_abs_det_jacobian (per sample).
#         """
#         # affine transform w^T z + b
#         w_dot_z = torch.matmul(z, self.w.unsqueeze(1)) + self.b  # (N, 1)
#         # tanh activation
#         activation = torch.tanh(w_dot_z)        
#         z_transformed = z + self.u * activation 
#         # jacobian determinant
#         phi_prime = 1 - activation**2
#         u_dot_w = torch.dot(self.u, self.w)     # scalar
#         # log determinant
#         log_abs_det = torch.log(torch.abs(1 + phi_prime.squeeze(1) * u_dot_w) + 1e-8)
#         return z_transformed, log_abs_det
 
 
 
class GeneTransformer(nn.Module):
    """
    Transformer-based autoencoder 
    """
    def __init__(
        self,
        in_dim=2000,       
        embed_dim=64,      
        latent_dim=512,    
        num_heads=8,
        num_layers=4,
        ff_factor=4,       
        decoder_hid=1024   
    ):
        super().__init__()
        
        self.gene_embedding = nn.Linear(1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_factor,
            batch_first=False  # by default it's (seq, batch, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.projection_head = nn.Linear(embed_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hid),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hid, in_dim), 
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G = x.shape
        
        # x_emb: (B, G, embed_dim)
        x_emb = self.gene_embedding(x.unsqueeze(-1))  # (B, G, 1) -> (B, G, embed_dim)
        
        x_emb = x_emb.permute(1, 0, 2)
        
        # Pass through Transformer: (G, B, embed_dim)
        encoded = self.transformer_encoder(x_emb)  # (G, B, embed_dim)
        
        encoded_mean = encoded.mean(dim=0)
        
        z = self.projection_head(encoded_mean)  # (B, latent_dim)
        
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_recon = self.decoder(z)
        return x_recon
    
  
class GeneMaskedAutoencoder(nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim, mask_ratio=0.2):
        super(GeneMaskedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x, decode:bool=False):
        # Masking
        mask = (torch.rand(x.shape) > self.mask_ratio).float().to(x.device)
        masked_x = x * mask
        
        # Encoding
        z = self.encoder(masked_x)
        return z
    
    def decode(self, z): 
        reconstructed_x = self.decoder(z)
        return reconstructed_x
    


def get_gene_decoder(decoder='linear',
                     embed_dim=256,
                     out_dim=250,
                     dropout=0.25,
                     n_layers=1,
                     num_head=8):

    if decoder == 'linear':
        model = LinearDecoder(embed_dim=embed_dim, out_dim=out_dim)
    elif decoder == 'mlp':
        model = MLPDecoder(embed_dim=embed_dim, 
                           dropout=dropout, 
                           out_dim=out_dim,
                           num_hid_dims=n_layers)
    elif decoder == 'transformer':
        model = TransformerDecoder(embed_dim=embed_dim, 
                                   n_head=num_head, 
                                   num_layers=n_layers, 
                                   dropout=dropout, 
                                   out_dim=out_dim)
    else:
        raise NotImplementedError(f"Gene decoder not implemented for {decoder}")
    
    return model


class LinearDecoder(nn.Module):
    def __init__(self, embed_dim=256, out_dim=250):
        super().__init__()
        self.predictor = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        out = self.predictor(x)
        return out
    

class MLPDecoder(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_hid_dims=0, 
                 dropout=0.1,
                 out_dim=250
                 ):
        super().__init__()

        hid_dims = [embed_dim] * num_hid_dims

        self.predictor = create_mlp(in_dim=embed_dim, 
                                    hid_dims=hid_dims, 
                                    act=nn.GELU(), 
                                    dropout=dropout,
                                    out_dim=out_dim, 
                                    end_with_fc=True)

    def forward(self, x):
        if len(x.shape) == 3: # Sequence of tokens
            x = x.mean(axis=1)

        out = self.predictor(x)
        return out


class TransformerDecoder(nn.Module):
    """
    Predict gene based on input tokens
    """
    def __init__(self, embed_dim=768, n_head=8, dropout=0.1, num_layers=1, out_dim=250):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_head, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Linear(embed_dim, out_dim)

    def reset_parameters(self):
        """
        Reinitializes parameters for all submodules within the TransformerDecoder to PyTorch defaults.
        """
        for layer in self.layers:
            layer._reset_parameters()  
        self.norm.reset_parameters()
        self.predictor.reset_parameters()


    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        for layer in self.layers:
            out, _ = layer(x, x, x)
            x = self.norm(x + out)
            
        out = x.permute(1, 0, 2).mean(axis=1)   # LND -> NLD -> ND

        out = self.predictor(out)

        return out


    
import torch
import torch.nn as nn

def gene_warmup(conf, 
                accelerator, 
                train_loader, 
                val_loader, 
                test_loader, 
                test_genes, 
                gene_model, 
                optimizer, 
                scheduler, 
                num_epochs=5, 
                ):
    
    
    
    # Define loss function and optimizer
    # get number of GPUs used during warmup
    
    # train_variance = train_loader.dataset.gene_adata.var.dispersions
    if conf['gene_loss'] in ['standardized_mse', 'barlow_std_mse']: 

        temp = pd.DataFrame(train_loader.dataset.gene_adata.X.toarray())
        mean = temp.mean(axis=0)
        std = temp.std(axis=0)
        gene_stats = pd.DataFrame({
            'mean': mean,
            'std': std,
        })
    else: 
        gene_stats = None
    
    
    loss_fn, _ = get_gene_loss_fn(method=conf['gene_loss'], world_size=1, gene_stats=gene_stats) # in main process, single GPU
    calc_metrics = CalcMetrics(metrics=['gene'])
    train_metrics = {k: AverageMeter(k, ':6.4f') for k in calc_metrics.keys()}
    val_metrics = {k: AverageMeter(k, ':6.4f') for k in calc_metrics.keys()}
    test_metrics = {k: AverageMeter(k, ':6.4f') for k in calc_metrics.keys()}
    
    

    gene_model = gene_model.to(accelerator.device)
    
    if conf['gene_model'] is None: 
        num_epochs = 1

    def log_metrics_dict(prefix: str, metrics_dict: Dict[str, AverageMeter], accelerator, iter):
        for k, v in metrics_dict.items(): 
            log_metric = {f"gene_warmup/{prefix}_{k}": v.avg}
            accelerator.log(log_metric, step=iter)
    
    for iter, epoch in enumerate(range(num_epochs)):
        # Training phase
            
        gene_model, train_loss, train_metrics = gene_train_epoch(conf, accelerator, gene_model, train_loader, optimizer, scheduler, loss_fn, calc_metrics, train_metrics, epoch)
        log_metrics_dict("train", train_metrics, accelerator, iter)
        losses = {'train': train_loss}
        if (epoch+1) % conf['warmup_val_every']== 0:
            if conf['separate_val_split']: 
                val_loss, val_metrics = gene_val_epoch(conf, accelerator, val_loader, gene_model, loss_fn, calc_metrics, test_metrics, epoch)
                log_metrics_dict("val", val_metrics, accelerator, iter)
                losses['val'] = val_loss
        if (epoch+1) % conf['warmup_test_every'] == 0:
            test_loss, test_metrics = gene_val_epoch(conf, accelerator, test_loader, gene_model, loss_fn, calc_metrics, test_metrics, epoch)
            log_metrics_dict("test", test_metrics, accelerator, iter)
            losses['test'] = test_loss
        
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for split, loss, in losses.items(): 
            print(f"{split} loss: {loss:.4f}")
            accelerator.log({f"gene_warmup/{split}_loss": loss}, step=iter)


        probe_results = gene_model_linprobe(
            train_loader, test_loader, 
                                            conf,
                                            test_genes=test_genes, 
                                            gene_model=gene_model, 
                                            accelerator=accelerator, 
                                            pca_reduce=conf['pca_reduce']
                                            )
        
        for organ, result in probe_results.items():
            log_result = {f"gene_warmup/{organ}_pearson_mean": result['pearson_mean'], f"gene_warmup/{organ}_pearson_std": result['pearson_std']}
            # if conf['warmup_only']: 
            accelerator.log(log_result, step=iter)
        # get average across organs
        avg_pearson = np.mean([result['pearson_mean'] for result in probe_results.values()])
        avg_std = np.mean([result['pearson_std'] for result in probe_results.values()])
        avg_spearman = np.mean([result['spearman_mean'] for result in probe_results.values()])
        avg_spearman_std = np.mean([result['spearman_std'] for result in probe_results.values()])

        log_result = {"gene_warmup/pearson_mean": avg_pearson, "gene_warmup/pearson_std": avg_std, "gene_warmup/spearman_mean": avg_spearman, "gene_warmup/spearman_std": avg_spearman_std}
        # if conf['warmup_only']: 
        accelerator.log(log_result, step=iter)
                        
    return gene_model
    

def gene_train_epoch(
            conf, 
            accelerator, 
            gene_model, 
            train_loader, 
            optimizer, 
            scheduler, 
            loss_fn: Callable,
            calc_metrics: Callable, 
            train_metrics: Dict, 
            epoch: int, 
        ): 
        
    print(f"Warmup Epoch {epoch+1}")
    gene_model.train()
    train_loss = 0.
    slide_pred_tracker = SlidePredTracker()
    
    pbar = tqdm(train_loader)
    for batch_idx, (x_img, gene_dict, meta) in enumerate(pbar):
        x_gene = gene_dict['X'].to(accelerator.device)
        if conf["warmup_noise"] > 0.0: 
            x_gene = add_noise(x_gene, method=conf["noise_method"], noise_factor=conf["warmup_noise"])
        
        model_loss_dict = None # optional model-specific loss
        
        if conf['gene_model'] == 'scgpt': 
            z_gene = gene_model(gene_dict)
        elif conf['gene_model'] == 'vae': 
            # z_gene, y_gene, mu, logvar, log_det = gene_model(x_gene)
            use_projection = True if conf['projection_head'] in ['linear', 'mlp'] else False
            ret_dict = gene_model(x_gene, projection=use_projection)
            # z_gene, y_gene = itemgetter('z', 'y')(ret_dict)
            z_gene, y_gene, mu, logvar, log_det = itemgetter('z', 'y', 'mu', 'logvar', 'total_log_det')(ret_dict)
            beta = (1 / z_gene.size(1)) if conf['kl_loss'] else 0.0
            model_loss_dict = gene_model.compute_loss(ret_dict, x_gene, beta=beta)
        else: 
            ret_dict = gene_model(x_gene)
            z_gene, y_gene = itemgetter('z', 'y')(ret_dict)
        
        
        loss = loss_fn(y_gene, x_gene) + sum(model_loss_dict.values())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        iter_loss = loss.item()
        train_loss += iter_loss * x_gene.size(0)
        train_metrics = calc_metrics(y_gene, x_gene, metrics='gene', metrics_dict=train_metrics)

        desc_str = f"Warmup Train {epoch+1}/{conf['warmup_epochs']}, {conf['gene_loss']} loss: {iter_loss:.4f}"
        metrics_str = ', '.join([
            f"{key:<10}: {value.avg:.5f}" for key, value in train_metrics.items()
        ])
        pbar.set_description(desc_str)
        pbar.set_postfix_str(metrics_str, refresh=True)
    
    
    train_loss /= len(train_loader.dataset)
    
    return gene_model, train_loss, train_metrics
        
        
def gene_val_epoch(
    conf, 
    accelerator, 
    split_loader, 
    gene_model, 
    loss_fn, 
    calc_metrics: Callable,
    split_metrics: Dict, 
    epoch, 
    split: str = 'val', 
): 

    gene_model.eval()
    split_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(split_loader)
        for x_img, gene_dict, meta in pbar:
            x_gene = gene_dict['X'].to(accelerator.device)
            
            model_loss_dict = {}
            
            if conf['gene_model'] == 'scgpt':
                z_gene = gene_model(gene_dict)
            elif conf['gene_model'] == 'vae': 
                ret_dict = gene_model(x_gene)
                z_gene, y_gene, mu, logvar, log_det = itemgetter('z', 'y', 'mu', 'logvar', 'total_log_det')(ret_dict)
                # inversely scale beta to latent size
                beta = (1 / z_gene.size(1)) if conf['kl_loss'] else 0.0
                model_loss_dict = gene_model.compute_loss(ret_dict, x_gene, beta=beta)
            else: 
                ret_dict = gene_model(x_gene)
                z_gene, y_gene = itemgetter('z', 'y')(ret_dict)
            
            loss = loss_fn(y_gene, x_gene) + sum(model_loss_dict.values())
            
            iter_loss = loss.item()
            split_loss += iter_loss * x_gene.size(0)
            split_metrics = calc_metrics(y_gene, x_gene, metrics='gene', metrics_dict=split_metrics)
            
            desc_str = f"Warmup {split.capitalize()} {epoch+1}/{conf['warmup_epochs']}, {conf['gene_loss']} loss: {iter_loss:.4f}"
            metrics_str = ', '.join([
            f"{key:<10}: {value.avg:.5f}" for key, value in split_metrics.items()
            ])
            pbar.set_description(desc_str)
            pbar.set_postfix_str(metrics_str, refresh=True)
        
    split_loss /= len(split_loader.dataset)
    return split_loss, split_metrics


def gene_model_linprobe(
    train_loader, 
    test_loader, 
    conf,
                        test_genes: Dict = None, 
                        gene_model: nn.Module = None, 
                        random_state: int = 42, 
                        accelerator: Accelerator=None,
                        pca_reduce:bool=True): 
    """
    Linear probe of gene model for gene_warmup pipeline
    """
    if gene_model is not None: 
        gene_model.eval()
    
    if accelerator is not None: 
        device = accelerator.device
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_test_genes = list(set(chain.from_iterable(test_genes.values())))
    
    test_gene_index = [test_loader.dataset.gene_adata.var.index.get_loc(g) for g in all_test_genes]
    gene_features = []
    gene_labels = []
    # all_gene_features = []
    # all_gene_labels = []
    
    def encode_features(loader): 
        with torch.no_grad(): 
            for _, batch_data in enumerate(tqdm(loader)): 
                _, gene_dict, _ = batch_data
                gene = gene_dict['X']
                gene = gene.to(device)
                if gene_model is None: 
                    z_gene = gene
                elif conf['gene_model'] == 'scgpt': 
                    z_gene = gene_model(gene_dict)
                elif conf['gene_model'] == 'vae': 
                    ret_dict = gene_model(gene)
                    z_gene, y_gene, mu, logvar, log_det = itemgetter('z', 'y', 'mu', 'logvar', 'total_log_det')(ret_dict)
                else: 
                    ret_dict = gene_model(gene)
                    z_gene, y_gene = itemgetter('z', 'y')(ret_dict)
                    # z_gene = gene_model(gene)
                gene_label = gene[:, test_gene_index]
                
                gene_features.append(z_gene.cpu().detach())
                gene_labels.append(gene_label.cpu().detach())
                del gene, gene_label, z_gene
        X = torch.cat(gene_features, axis=0).numpy()
        y = torch.cat(gene_labels, axis=0).numpy()
        
        return X, y
    
    Z_train, y_train = encode_features(train_loader)
    Z_test, y_test = encode_features(test_loader)
    
    probe_results = run_linprobe(Z_train, Z_test, y_train, y_test, test_genes=test_genes, pca_reduce=pca_reduce, random_state=random_state, model=None, test_gene_index=None)
    return probe_results
    


def plot_performance_by_gene(): 
    pass


def add_noise(batch, method: str="random", noise_factor: float = 0.1): 
    if method == "random": 
        batch = _add_random_masking_noise(batch, noise_factor)
    elif method == "gaussian": 
        batch = _add_gaussian_noise(batch, noise_factor)
    
    return batch

def _add_random_masking_noise(batch: torch.tensor, noise_factor: float = 0.1): 
    mask = torch.rand_like(batch) > noise_factor
    noisy_x = batch * mask.float()
    return noisy_x

def _add_gaussian_noise(batch: torch.tensor, noise_std: float = 0.1): 
    noise = torch.randn_like(batch) * noise_std
    noisy_batch = batch + noise
    noisy_batch = torch.clamp(noisy_batch, min=0.0) # ensure non-negative
    return noisy_batch


############################
# scGPT
############################

# script_directory = os.path.dirname(os.path.abspath(__file__))
# if script_directory not in sys.path:
#     sys.path.append(script_directory)
# from scgpt.model import TransformerModel
# from scgpt.utils import load_pretrained


# from torchtext.vocab import Vocab
# from torchtext._torchtext import (  #It requires python version <=3.11
#     Vocab as VocabPybind,
# )
# from scgpt.tokenizer.gene_tokenizer import GeneVocab
from pathlib import Path




class SCGPTEncoder(nn.Module):

    def __init__(self, pretrained,linear_layer=False,embed_dim=512, n_genes: int=250):     
        super().__init__()
        
        model_dir = os.path.dirname(pretrained)
        model_dir = Path(model_dir)
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file) 
        special_tokens = ["<pad>", "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])

        ntokens = len(vocab)
        model = TransformerModel(ntoken=ntokens, # length of the vocabulary (60697) 
                                d_model=512, #embedding size
                                nhead=8,
                                d_hid=512,
                                nlayers=12,
                                nlayers_cls=3, 
                                vocab=vocab, 
                                dropout=0.2,
                                pad_token="<pad>",
                                pad_value=-2, # (for continuous encoder) and 51 for categorical encoder
                                do_dab=False,
                                use_batch_labels=False, # Use batch labels for batch correction with adversarial training
                                domain_spec_batchnorm=False,
                                n_input_bins=51, #51 (for continous encoder) and 53 for categorical encoder
                                pre_norm=False,
                                use_fast_transformer=True,
                                explicit_zero_prob=True,
                                ecs_threshold=0.8,
                                do_mvc=True, 
                                num_batch_labels=None) # Required for adversarial training of batch correction
        
        load_pretrained(model, torch.load(pretrained), verbose=False) # everything is trainable by default

        if linear_layer:
            linear_layer = nn.Linear(512, embed_dim)  
            model.linear_layer = linear_layer

        self.model = model
        
        self.decoder = get_gene_decoder(decoder='mlp', 
                                        embed_dim=embed_dim, 
                                        out_dim=n_genes
                                        )
        
    def forward(self, x_gene: Tuple, agg_scgpt='cls'): 
        
        (scgpt_genes, scgpt_values) = x_gene # (b, n_genes) and (b, n_genes)
        
        with autocast():
            gene_embed = self.model._encode(src=scgpt_genes,
                                values=scgpt_values,
                                src_key_padding_mask=None,
                                batch_labels = None)
        
        if agg_scgpt == 'cls':
            gene_embed = self.model._get_cell_emb_from_layer(layer_output=gene_embed) # get the <cls> token
        elif agg_scgpt == 'mean_token':
            gene_embed = torch.mean(gene_embed[:,1:,:],axis=1)
        elif agg_scgpt == 'mean_cls_and_tokens':
            cls_tok = self.model._get_cell_emb_from_layer(layer_output=gene_embed)
            mean_tok = torch.mean(gene_embed[:,1:,:],axis=1)
            gene_embed = (cls_tok + mean_tok) / 2
        else:
            print(f'{agg_scgpt} not implemented for agg_scgpt')
        
        if hasattr(self.model, 'linear_layer'):
            gene_embed = self.model.linear_layer(gene_embed)    

        
        return gene_embed
        
        
        
    


def unfreeze_scgpt(model,n_components=0):
    """
    Partially freeze the encoder.

    Args:
    - encoder (str): Encoder name
    - n_components (int): Number of blocks to unfreeze (from the output)
    """

    if n_components < 0:    # Unfreeze every parameter
        for k, v in model.named_parameters():
            v.requires_grad = True
    else:
        for k, v in model.named_parameters():
            v.requires_grad = False

        assert n_components < 12
        num_list = [ 11 - idx for idx in range(n_components)]
        incl_list = [f'transformer_encoder.layers.{num}' for num in num_list] 
        for k, v in model.named_parameters():
            for name in incl_list:
                if name in k:
                    v.requires_grad = True
                    break
        if hasattr(model, 'linear_layer'):
            for k, v in model.linear_layer.named_parameters():
                v.requires_grad = True
            
    return model




###############
# scGPT utils
###############

import pandas as pd
import numpy as np
import glob as glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse
# from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
# from scgpt.tokenizer.gene_tokenizer import GeneVocab
# from scgpt.preprocess import Preprocessor
from typing import *


def check_gene_vocab(adata: AnnData, scgpt_loc: str): 
    
    adata = adata.copy()
    adata.var["gene_name"] = adata.var.index.tolist()
    
    # Find common genes with vocabulary
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    model_dir = Path(scgpt_loc)
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    return adata, vocab


def prepare_data_scGPT(adata0,
                       model_loc, 
                       use_key: Optional[str] = None,
                       filter_gene_by_counts: Union[int, bool] = False,
                       filter_cell_by_counts: Union[int, bool] = False,
                       normalize_total: Union[float, bool] = False,
                       result_normed_key: Optional[str] = "X_normed",
                       log1p: bool = False,
                       result_log1p_key: str = "X_log1p",
                       subset_hvg: Union[int, bool] = False,
                       hvg_use_key: Optional[str] = None,
                       hvg_flavor: str = "seurat_v3",
                       binning: Optional[int] = None,
                       result_binned_key: str = "X_binned",
                       **kwargs,
                       ):

    print(f"Preparing data for scGPT")
    # Set gene_name and batch_id variables
    adata = adata0.copy()
    adata.var["gene_name"] = adata.var.index.tolist()
    data_is_raw = False if 'log1p' in adata.uns else True
 
    adata, vocab = check_gene_vocab(adata, model_loc)

    # set up the preprocessor (finetuning)
    # n_hvg=1200
    if not subset_hvg:
        n_hvg = len(adata.var)
    else: 
        n_hvg = subset_hvg 
        
    preprocessor = Preprocessor(
        use_key=use_key,  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=filter_cell_by_counts,  # step 2
        normalize_total=normalize_total,  # 3. whether to normalize the raw data and to what sum
        result_normed_key=result_normed_key,  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data # TODO - false
        result_log1p_key=result_log1p_key,
        subset_hvg=subset_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=binning,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key=result_binned_key,  # the key in adata.layers to store the binned data
    )
    preprocessor(adata)

    
    # Tokenize the input data for model finetuning
    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    # Variables
    pad_value = -2
    pad_token = "<pad>"
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    max_seq_len = n_hvg + 1
    # max_seq_len = n_hvg
    
    tokenized = tokenize_and_pad_batch(all_counts,
                                       gene_ids, #number --> each number indicates one gene
                                       max_len=max_seq_len,
                                       vocab=vocab,
                                       pad_token=pad_token,
                                       pad_value=pad_value,
                                       append_cls=True,  # append <cls> token at the beginning
                                       include_zero_gene=True, 
                                       )

    adata.obsm['scgpt_genes'] = np.array(tokenized['genes'])
    adata.obsm['scgpt_values'] = np.array(tokenized['values']) # binned values
    
    return adata
