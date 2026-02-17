import torch.nn as nn


class VisionAdapter(nn.Module): 
    """
    Simple fine-tuning head that takes in a vision model adds some linear layers producing the 
    same dimensions

    """
    def __init__(self, vision_enc: nn.Module, dims: int = 512, bottleneck: int = None): 
        """
        Args:
            vision_enc (nn.Module): vision encoder that is to be fine-tuned
        """
        super().__init__()
        self.bottleneck = int(bottleneck)
        self.encoder = vision_enc
        for param in self.encoder.parameters(): 
            param.requires_grad = False
        self.encoder.eval()
        
    
        if bottleneck is None: 
            self.adapter = nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.LayerNorm(dims),
                nn.Linear(dims, dims),
                nn.LayerNorm(dims)
            )
        else: 
            self.adapter = nn.Sequential(
                nn.Linear(dims, self.bottleneck), 
                nn.ReLU(), 
                nn.Linear(self.bottleneck, dims)
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.adapter(x)
        return x
        
