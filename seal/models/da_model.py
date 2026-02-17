"""
Model utils for domain adaptation loss (gradient reversal layer)
"""
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)

class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        z_dim: int, # embedding dimensions of the encoder
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = True,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(z_dim, z_dim))
            self._decoder.append(nn.LayerNorm(z_dim))
            self._decoder.append(activation())
        self.out_layer = nn.Linear(z_dim, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x, lambd):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=lambd) 
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)