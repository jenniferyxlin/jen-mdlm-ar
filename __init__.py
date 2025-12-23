from .model import MDLModel, RotaryPositionalEmbedding
from .diffusion import get_masked_batch, masked_cross_entropy_loss

__version__ = "0.2.0"
__all__ = ['MDLModel', 'RotaryPositionalEmbedding', 'get_masked_batch', 'masked_cross_entropy_loss']

from .model import ARModel, RotaryPositionalEmbedding
from .ar import get_ar_batch, ar_cross_entropy_loss

__version__ = "0.1.0"
__all__ = ['ARModel', 'RotaryPositionalEmbedding', 'get_ar_batch', 'ar_cross_entropy_loss']