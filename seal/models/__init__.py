"""
SEAL Models Module

This module contains model architectures and loading utilities.
"""

from seal.models.encoder_factory import encoder_factory, seal_factory


def load_model(backbone:str, baseline:bool=False):
    """
    Load a SEAL encoder model.

    Args:
        encoder_name (str): Name of the encoder to load.
            Supported encoders: 'conch', 'uni', 'univ2', 'phikon', 'phikon2',
            'resnet50', 'gigapath', 'virchow', 'virchow2', 'h0mini',
            'hoptimus0', 'hoptimus1', 'musk'

    Returns:
        InferenceEncoder: The loaded encoder model instance with:
            - model: The underlying neural network
            - eval_transforms: Preprocessing transforms for the model
            - precision: Recommended precision (torch.float32, torch.float16, etc.)

    Example:
        >>> from seal import load_model
        >>> encoder = load_model("conch")
        >>> # Use encoder.eval_transforms to preprocess images
        >>> # Use encoder(x) to get embeddings
    """
    # img model
    
    if baseline: 
        img_model = encoder_factory(backbone)
    else: 
        # 
        img_model = seal_factory(backbone)
    
    # gene model 
    
    return img_model


__all__ = ['load_model', 'encoder_factory']
