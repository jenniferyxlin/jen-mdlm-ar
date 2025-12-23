"""
Autoregressive batch preparation and loss functions - simplified version of pretrain_gpt.py
"""

import torch
import torch.nn.functional as F


def get_ltor_masks_and_position_ids(tokens, eod_token=None, eod_mask_loss=True):
    """
    Create loss masks and position ids for autoregressive training.
    Excludes EOD tokens from loss computation.
    
    Args:
        tokens: [batch_size, seq_len] - input tokens
        eod_token: End-of-document token ID (optional)
        eod_mask_loss: If True, mask out EOD tokens from loss
    
    Returns:
        loss_mask: [batch_size, seq_len] - 1.0 for valid tokens, 0.0 for EOD
        position_ids: [batch_size, seq_len] - position indices
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Initialize loss mask to 1.0, zero out EOD positions 
    loss_mask = torch.ones((batch_size, seq_len), dtype=torch.float, device=device)
    if eod_mask_loss and eod_token is not None:
        loss_mask[tokens == eod_token] = 0.0
    
    # Generate position ids for each sample
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    return loss_mask, position_ids


def get_ar_batch(tokens, eod_token=None, eod_mask_loss=True, randmask_ratio=0.0, eps=1e-3):
    """
    Prepare batch for autoregressive training.
    Shifts tokens to create input/label pairs for next-token prediction.
    
    Args:
        tokens: [batch_size, seq_len+1] - input tokens (includes target token)
        eod_token: End-of-document token ID (optional)
        eod_mask_loss: If True, mask out EOD tokens from loss
        randmask_ratio: Probability of applying random masking to attention (0.0 = no random masking)
        eps: Minimum mask probability for random masking (used when randmask_ratio > 0)
    
    Returns:
        input_ids: [batch_size, seq_len] - input tokens (without last token)
        labels: [batch_size, seq_len] - target tokens (shifted by 1)
        loss_mask: [batch_size, seq_len] - loss mask (0.0 for EOD tokens)
        attention_mask: [batch_size, seq_len, seq_len] - attention mask (True = mask out)
        position_ids: [batch_size, seq_len] - position indices
    """
    batch_size, seq_len_plus_one = tokens.shape
    device = tokens.device
    seq_len = seq_len_plus_one - 1
    
    # Shift tokens: input = tokens[:-1], labels = tokens[1:]
    input_ids = tokens[:, :-1].contiguous()  # [batch_size, seq_len]
    labels = tokens[:, 1:].contiguous()      # [batch_size, seq_len]
    
    # Get loss mask and position ids
    loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids,
        eod_token=eod_token,
        eod_mask_loss=eod_mask_loss
    )
    
    # Create attention mask
    # Start with causal mask (upper triangular = True means mask out)
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    # Expand to batch dimension: [batch_size, seq_len, seq_len]
    attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply random masking if requested (similar to Megatron's randmask_ratio)
    if randmask_ratio > 0:
        for i in range(batch_size):
            # Randomly select tokens to mask
            rand_toks = torch.rand(seq_len, device=device) < randmask_ratio
            
            # Create probability mask for attention
            t = torch.rand(seq_len, device=device)
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask.repeat(seq_len, 1).permute(1, 0)  # [seq_len, seq_len]
            
            # Create additional attention mask based on probabilities
            additional_attn_mask = (torch.rand((seq_len, seq_len), device=device) < p_mask) & rand_toks.unsqueeze(1)
            
            # Combine with causal mask (OR operation: mask if either is True)
            attention_mask[i, :, :] = attention_mask[i, :, :] | additional_attn_mask
            
            # Remove diagonal: tokens can always attend to themselves
            attention_mask[i, :, :] = attention_mask[i, :, :] & ~torch.eye(seq_len, device=device, dtype=torch.bool)
    
    return input_ids, labels, loss_mask, attention_mask, position_ids


def ar_cross_entropy_loss(logits, labels, loss_mask):
    """
    Compute standard masked cross-entropy loss for autoregressive models.
    Matches pretrain_gpt.py loss calculation:
    - Computed on all valid tokens (excluding padding/EOD)
    - Averaged by the number of valid tokens
    - Uses causal attention (left-to-right)
    
    Only computes loss on valid tokens (excludes EOD tokens).
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model predictions
        labels: [batch_size, seq_len] - target tokens
        loss_mask: [batch_size, seq_len] - loss mask (0.0 for EOD tokens)
    
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
    losses = losses * loss_mask
    
    # Average over valid tokens
    valid_token_count = loss_mask.sum().float()
    loss = losses.sum() / (valid_token_count + 1e-8)
    
    return loss
