"""
Sanity check to verify that MDLM forward pass works. 
"""

import torch
from model import MDLModel
from diffusion import get_masked_batch, masked_cross_entropy_loss

# Create a small model
model = MDLModel(
    vocab_size=1000,
    d_model=128,
    n_layers=2,
    n_heads=4,
    d_ff=None, # hard-coded
    max_seq_len=64
)

# Create dummy input
batch_size = 4
seq_len = 32
tokens = torch.randint(0, 1000, (batch_size, seq_len))

print("Original tokens shape:", tokens.shape)
print("Sample tokens:", tokens[0, :10])

# Apply masked diffusion forward process
noisy_input, labels, EOD_mask, masked_indices, p_mask, position_ids = get_masked_batch(
    tokens,
    eps=1e-3,
    mask_token_id=model.mask_token_id
)

print("\nAfter masking...")
print("Noisy input shape:", noisy_input.shape)
print("Masked positions:", masked_indices[0, :10])
print("Mask probabilities:", p_mask[0, :10])
print("Sample noisy input:", noisy_input[0, :10])

# Forward pass
logits = model(noisy_input)
print("\nModel output shape:", logits.shape)

# Compute loss
loss = masked_cross_entropy_loss(logits, labels, EOD_mask, masked_indices, p_mask)
print("Loss:", loss.item())

print("\n✓ MDM forward pass successful:)")