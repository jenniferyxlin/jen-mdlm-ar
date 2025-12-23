"""
Masked diffusion forward process to connect diffusion models (continuous noise) to discrete tokens sequences. 
Treat masking as a discrete diffusion process where each step replaces a token with a mask token. 
"""

import torch
import torch.nn.functional as F
import math


def get_ltor_masks_and_position_ids(tokens, eod_token=None, eod_mask_loss=True):
    """
    Create loss masks and position ids to exclude EOD tokens from loss. 
    
    Args:
        tokens: [batch_size, seq_len] - input tokens
        eod_token: End-of-document token ID (optional)
        eod_mask_loss: If True, mask out EOD tokens from loss
    
    Returns:
        EOD_mask: [batch_size, seq_len] - 1.0 for valid tokens, 0.0 for EOD
        position_ids: [batch_size, seq_len] - position indices
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Initialize loss mask to 1.0, zero out EOD positions 
    EOD_mask = torch.ones((batch_size, seq_len), dtype=torch.float, device=device)
    if eod_mask_loss and eod_token is not None:
        EOD_mask[tokens == eod_token] = 0.0
    
    # Generate position ids for each sample
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    return EOD_mask, position_ids


def get_masked_batch(tokens, eps=1e-3, mask_token_id=None, device=None, eod_token=None, mask_schedule='linear'):
    """
    Implement masked diffusion forward process by randomly masking tokens. 
    
    Args:
        tokens: [batch_size, seq_len] - original tokens
        eps: minimum mask probability (default 1e-3)
        mask_token_id: ID to use for masked tokens
        device: device for tensors
        eod_token: End-of-document token ID (optional)
        mask_schedule: 'linear' or 'cosine' - masking schedule type (default 'linear')
    
    Returns:
        noisy_input: [batch_size, seq_len] - tokens with some replaced by mask_token_id
        labels: [batch_size, seq_len] - original tokens (targets)
        EOD_mask: [batch_size, seq_len] - loss mask (0.0 for EOD tokens)
        masked_indices: [batch_size, seq_len] - boolean mask of which positions were masked
        p_mask: [batch_size, seq_len] - mask probability for each position
        position_ids: [batch_size, seq_len] - position indices
    """
    if device is None:
        device = tokens.device
    
    batch_size, seq_len = tokens.shape
    
    # Remove last token 
    tokens = tokens[:, :-1].contiguous()
    seq_len = tokens.shape[1]
    
    # Sample noise level t ~ Uniform(0, 1) for each sample (each sample has its own noise level)
    t = torch.rand(batch_size, device=device)
    
    # Compute mask probability based on schedule
    if mask_schedule == 'linear': # p_mask = (1 - eps) * t + eps
        # Ensures minimum masking of eps and maximum of 1.0
        p_mask = (1 - eps) * t + eps
    elif mask_schedule == 'cosine': #p_mask = eps + (1 - eps) * (1 - cos(π * t / 2))
        # Smooth transition with slower start and faster end
        p_mask = eps + (1 - eps) * (1 - torch.cos(math.pi * t / 2))
    else:
        raise ValueError(f"Unknown mask_schedule: {mask_schedule}. Needs to be 'linear' or 'cosine'")
    
    p_mask = p_mask[:, None].expand(-1, seq_len)  # [batch_size, seq_len]
    
    # Randomly mask tokens based on p_mask
    masked_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
    
    # Replace masked tokens with mask_token_id
    if mask_token_id is None:
        # Use vocab_size as mask token (assuming tokens are in range [0, vocab_size-1])
        mask_token_id = tokens.max().item() + 1
    
    noisy_input = torch.where(masked_indices, mask_token_id, tokens)
    labels = tokens  # Original tokens are the targets
    
    # Get loss mask and position ids
    EOD_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token=eod_token,
        eod_mask_loss=True
    )
    
    return noisy_input, labels, EOD_mask, masked_indices, p_mask, position_ids 


def masked_cross_entropy_loss(logits, labels, EOD_mask, masked_indices, p_mask):
    """
    Compute importance-weighted masked cross-entropy loss to get unbiased loss estimates over all positions.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model predictions
        labels: [batch_size, seq_len] - target tokens
        EOD_mask: [batch_size, seq_len] - loss mask (0.0 for EOD tokens)
        masked_indices: [batch_size, seq_len] - boolean mask
        p_mask: [batch_size, seq_len] - mask probabilities
    
    Returns:
        loss: scalar tensor
    """
    # Standard cross-entropy loss
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none'
    ).view(labels.shape)  # [batch_size, seq_len]
    
    # Apply loss mask (excludes EOD tokens)
    losses = losses * EOD_mask
    
    # Only compute loss on masked positions
    masked_losses = losses[masked_indices]
    masked_p = p_mask[masked_indices]
    
    # Weight by inverse mask probability
    weighted_losses = masked_losses / masked_p
    
    # Count valid masked positions (masked AND not EOD tokens)
    valid_masked_count = (masked_indices & (EOD_mask > 0)).sum().float()
    
    # Average over valid masked positions (add small epsilon to avoid division by zero)
    loss = weighted_losses.sum() / (valid_masked_count + 1e-8)
    
    return loss