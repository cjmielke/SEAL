import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# import torchsort
from seal.utils.loss_utils import gather_features

def get_gene_loss_fn(method: str, world_size: int = 1, gene_stats:pd.DataFrame=None) -> nn.Module:
    
    print(f"Using {method.upper()} gene loss")
    
    gene_loss_dict = { 
                      'mse': {
                          'loss': nn.MSELoss(), 
                          'scale_factor': 10.0},
                      'standardized_mse': {
                          'loss': StandardizedMSELoss(gene_stats=gene_stats, world_size=world_size),
                          'scale_factor': 10.0,
                        }, 

                      'pcc': { # per-sample pearson
                          'loss': PearsonSampleLoss(agg='mean', world_size=world_size),
                          'scale_factor': 5.0
                          }, 
                      'pls': {
                          'loss': DeepPLSLoss(), 
                          'scale_factor': 1.0}, 
                      'barlow': {
                          'loss': BarlowTwinsLoss(world_size=world_size), 
                          'scale_factor': 0.1}, # correct magnitude to CLIP loss
                      'barlow_soft': {
                          'loss': BarlowTwinsLoss(lambd=0.0, world_size=world_size), 
                          'scale_factor': 0.1},
                      'barlow_mse': {
                          'loss': BarlowMSELoss(lambd=0.0, world_size=world_size, mse_norm=False), 
                          'scale_factor': 0.1}, 
                      'barlow_std_mse': {
                          'loss': BarlowMSELoss(lambd=0.0, mse_norm=True, gene_stats=gene_stats, world_size=world_size),
                          'scale_factor': 0.1},

                      'mmd': {
                          'loss': MMDLoss(), 
                          'scale_factor': 1.0}, 
                      'l1': {
                          'loss': nn.L1Loss(), 
                          'scale_factor': 10.0}, 
                      'huber': {
                          'loss': nn.SmoothL1Loss(), 
                          'scale_factor': 10.0}, 
                      'negbin': {
                          'loss': NegBinomialLoss(), 
                          'scale_factor': 10.0}, 
    }

    loss, scale_factor = gene_loss_dict[method].copy().values()
    
    return loss, scale_factor



class StandardizedMSELoss(nn.Module): 
    
    def __init__(self, gene_stats: pd.DataFrame=None, world_size: int=1): 
        """
        Standardized MSE loss
        """
        super().__init__()
        self.world_size = world_size
        
        if gene_stats is not None: 
            means = torch.tensor(gene_stats['mean'].values, dtype=torch.float32)
            stds = torch.tensor(gene_stats['std'].values, dtype=torch.float32)
    
            self.register_buffer('means', means)
            self.register_buffer('stds', stds)
        self.eps = 1e-8 # prevent div by 0
        
        
    def forward(self, y: torch.tensor, x: torch.tensor):
        
        if self.means.device != y.device: 
            self.means = self.means.to(y.device)
            self.stds = self.stds.to(y.device)
        
        # normalized using mean and std per gene from the train set
        y_s = (y - self.means) / (self.stds + self.eps)
        x_s = (x - self.means) / (self.stds + self.eps)

        return F.mse_loss(y_s, x_s, reduction='mean')


class PearsonSampleLoss(nn.Module): 
    def __init__(self, agg: str='mean', world_size:int=1): 
        """
        L = 1-spearman rank correlation
        """
        super().__init__()
        self.agg = agg
        self.world_size = world_size
        
    def forward(self, y: torch.tensor, x: torch.tensor): 
        # per-sample pearson correlation
        xm = x - x.mean(-1, keepdim=True)
        ym = y - y.mean(-1, keepdim=True)
        num = (xm * ym).sum(-1)
        den = torch.sqrt((xm**2).sum(-1) * (ym**2).sum(-1)) + 1e-8
        corr = num / den
        loss = 1 - corr.mean() 
        return loss
        


class BarlowMSELoss(nn.Module): 
    def __init__(self, lambd, mse_norm: bool = False, gene_stats: pd.DataFrame=None, 
                 world_size=1): 
        """
        Minimal wrapper for Barlow + MSE loss
        """
        super().__init__()
        self.lambd = lambd
        
        # if mse_norm: 
        #     assert gene_stats is not None, "gene_stats must be provided for standardized MSE loss"
        
        self.barlow = BarlowTwinsLoss(lambd=lambd, world_size=world_size)
        if mse_norm: 
            self.mse = StandardizedMSELoss(gene_stats=gene_stats, world_size=world_size)
            self.lambda_mse = 5.0
        else: 
            self.mse = nn.MSELoss()
            self.lambda_mse = 1.0
        
        self.lambda_barlow = 1.0
        
    def forward(self, Z1, Z2): 
        # Z1, Z2: [batch_size, d]
        loss = self.lambda_barlow * self.barlow(Z1, Z2) + self.lambda_mse * self.mse(Z1, Z2)
        return loss


class MMDLoss(nn.Module):
	
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, kernel='rbf'):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.kernel = kernel
        self.fix_sigma = None


    def forward(self, x, y): 
        return self.mmd(x, y)

    def mmd(self, x, y):
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # transpose initial implementation for gene-wise MMD
        xx, yy, zz = torch.mm(x.t(), x), torch.mm(y.t(), y), torch.mm(x.t(), y) # (n_genes x n_genes)
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device))

        if self.kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1

        if self.kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)

        return torch.mean(XX + YY - 2. * XY)


class BarlowTwinsLoss(nn.Module): 
    def __init__(self, 
                 lambd=5e-3, 
                 local_loss=False,
                 gather_with_grad=False,
                 cache_labels=False,
                 rank=0,
                 world_size=1,
                 use_horovod=False,
                 scaling_factor=0.05, 
                 ): 
        super().__init__()
        self.lambd = lambd
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size # no. GPUs
        self.use_horovod = use_horovod
        self.local_loss = local_loss
        self.scaling_factor = scaling_factor
 
    def forward(self, Z1, Z2):
        # Z1, Z2: [batch_size, d]
        
        if self.world_size > 1: 
            Z1, Z2 = gather_features(Z1, Z2, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
        
        B, D = Z1.shape

        # Center
        Z1 = Z1 - Z1.mean(dim=0, keepdim=True)
        Z2 = Z2 - Z2.mean(dim=0, keepdim=True)

        # Normalize
        Z1 = Z1 / (Z1.var(dim=0, unbiased=False, keepdim=True).add(1e-9).sqrt())
        Z2 = Z2 / (Z2.var(dim=0, unbiased=False, keepdim=True).add(1e-9).sqrt())

        # Cross-correlation
        c = (Z1.T @ Z2) / B  # [d, d]

        #   sum_i (1 - c_ii)^2 + lambd * sum_{i != j} c_{ij}^2
        diag = torch.eye(D, device=c.device)
        c_diff = c - diag
        c_diff_sq = c_diff.pow(2)

        loss = self.lambd * c_diff_sq.sum() + (1 - self.lambd) * c_diff_sq.diagonal().sum()
        loss *= self.scaling_factor
        return loss


class NegBinomialLoss(nn.Module): 
    def __init__(self): 
        super().__init__()
        
    
    def forward(self, z1, z2, alpha=0.1, eps=1e-8):
        """
        z1, z2: [batch_size, n_genes] in log1p space.
        alpha: Dispersion parameter (fixed or learnable).
        eps: small constant for numerical stability.
        """
        alpha_t = torch.as_tensor(alpha, dtype=z1.dtype, device=z1.device)

        mu = torch.expm1(z1).clamp(min=eps)
        z2 = torch.expm1(z2).clamp(min=0.0)

        p = mu / (mu + alpha_t)

        log_prob = (
            torch.lgamma(z2 + alpha_t)
            - torch.lgamma(alpha_t)
            - torch.lgamma(z2 + 1.0)
            + alpha_t * torch.log(1 - p + eps)
            + z2 * torch.log(p + eps)
        )
        return -log_prob.mean()


class DeepPLSCorrLoss(nn.Module): 
    def __init__(self, epsilon=1e-12):
        super(DeepPLSCorrLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, Z, Y): 
        """
        Computes an average correlation across all dimension pairs (z_i, y_j).
        This is not strictly partial least squares, but a dimensionless measure
        that won't explode in scale.

        Z: [batch_size, d_Z]
        Y: [batch_size, d_Y]
        """
        B = Z.shape[0]

        Z_mean = Z.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Z_centered = Z - Z_mean
        Y_centered = Y - Y_mean

        dot_ij = Z_centered.transpose(0, 1) @ Y_centered  # shape [d_Z, d_Y]

        z_norm = torch.norm(Z_centered, dim=0, keepdim=True) + self.epsilon  # [1, d_Z]
        y_norm = torch.norm(Y_centered, dim=0, keepdim=True) + self.epsilon  # [1, d_Y]

        correlation_mat = dot_ij / (z_norm.T @ y_norm)

        mean_corr = correlation_mat.mean()
        loss = -mean_corr
        return loss
    
    

class DeepPLSLoss(nn.Module): 
    def __init__(self, alpha=1e-4):
        super(DeepPLSLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, Z, Y):
        """
        A 'scaled' version of the PLS-style loss that uses the Frobenius norm
        of the cross-covariance but multiplies by a small alpha to keep 
        values in a more stable range.

        Z: [batch_size, d_Z]
        Y: [batch_size, d_Y]
        alpha: scaling factor (tune as needed)
        """
        # 1) Center Z and Y
        Z_mean = Z.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Z_centered = Z - Z_mean
        Y_centered = Y - Y_mean

        # 2) Cross-covariance (d_Z x d_Y)
        B = Z.shape[0]
        cov_zy = (Z_centered.T @ Y_centered) / (B - 1)

        # 3) Frobenius norm of cross-covariance
        cov_zy_frob = torch.norm(cov_zy, p='fro')

        # 4) Scale by alpha and invert sign to "maximize" covariance
        loss = -self.alpha * cov_zy_frob
        return loss