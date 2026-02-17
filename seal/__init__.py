"""
SEAL: Spatial Expression-Aligned Learning

Vision-omics model for histology and spatial transcriptomics inference.
"""

import warnings

# Third-party deprecation noise during model imports.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata\.utils")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The cuda\..* module is deprecated.*",
)

from seal.models import load_model, seal_factory
from seal.omics_preprocess import preprocess_spot

__version__ = "0.0.1"
__all__ = ['load_model', 'seal_factory', 'preprocess_spot']
