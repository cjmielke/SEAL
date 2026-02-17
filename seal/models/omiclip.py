###### OmiCLIP model

import torch.nn as nn
from torchvision import transforms

import loki.predex
import loki.utils
import loki.preprocess


class omiclip_img(nn.Module):
    def __init__(self, model, **kwargs): 
        super().__init__()
        self.encoder = model
        self.decoder = None

    def forward(self, x, decode=False):
        
        z = self.encoder.encode_image(x)

        return z

class omiclip_text(nn.Module):
    def __init__(self, model, tokenizer, **kwargs): 
        super().__init__()
        self.encoder = model
        self.tokenizer = tokenizer

    def forward(self, x):
        device = next(self.parameters()).device

        # Tokenizer returns a dict of tensors
        text_inputs = self.tokenizer(x)
        z = self.encoder.encode_text(text_inputs.to(device))

        return {'z': z}

def omiclip_encoder(model_path, device='cpu'):
    img_encoder = None
    gene_encoder = None

    model, _, tokenizer = loki.utils.load_model(model_path, device)

    img_encoder = omiclip_img(model)
    gene_encoder = omiclip_text(model, tokenizer)   # OmiCLIP gene encoder is basically a text processor

    return img_encoder, gene_encoder