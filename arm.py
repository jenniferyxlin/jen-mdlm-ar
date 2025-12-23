"""
Basic AR (Autoregressive) model implementation - simplified version of pretrain_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotary positional embedding to tensor.
    t: [seq_len, batch, ..., dim] or [batch, seq_len, ..., dim]
    freqs: [seq_len, ..., dim] - rotary embedding frequencies
    """
    # Handle batch-first or sequence-first formats
    if t.dim() == 3 and t.shape[0] != freqs.shape[0]:
        # Assume batch-first: [batch, seq_len, dim]
        t = t.transpose(0, 1)  # [seq_len, batch, dim]
        transpose_back = True
    else:
        transpose_back = False
    
    rot_dim = freqs.shape[-1]
    t_pass = None
    if t.shape[-1] != rot_dim:
        # Partial rotary embedding to first half of the hidden dims
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    
    freqs_ = freqs[:t.shape[0]]
    cos = freqs_.cos().to(t.dtype)
    sin = freqs_.sin().to(t.dtype)
    
    # Apply rotary embedding: t * cos + rotate(t) * sin
    t_rotated = (t * cos) + (_rotate_half(t) * sin)
    
    # Concatenate the hidden dims if partial RoPE was applied
    if t_pass is not None:
        t_rotated = torch.cat((t_rotated, t_pass), dim=-1)
    
    if transpose_back:
        t_rotated = t_rotated.transpose(0, 1)
    
    return t_rotated


class RMSNorm(nn.Module):
    """
    Normalizes the input by dividing by the root mean square of the input.
    Skips mean-centering in LayerNorm.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


def swiglu(gate, up):
    """
    SwiGLU activation function: Swish(gate) * up
    Swish(x) = x * sigmoid(x) = silu(x)
    Args:
        gate: [..., d_ff] - gate projection output
        up: [..., d_ff] - up projection output
    Returns:
        [..., d_ff] - SwiGLU output
    """
    return F.silu(gate) * up  # silu is Swish


class RotaryPositionalEmbedding(nn.Module):
    """Generates rotation frequencies for RoPE. 
    Precomputes inverse frequencies and produces cos/sin values used to rotate Q/K vectors given position ids.
    """
    
    def __init__(self, dim, theta=10000.0, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.theta = theta # base frequency
        self.max_seq_len = max_seq_len 
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, position_ids):
        """
        Generate rotary embeddings for given position ids.
        position_ids: [batch_size, seq_len] or [seq_len]
        Returns: [seq_len, 1, dim] - rotary embedding frequencies
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        seq_len = position_ids.shape[1]
        device = position_ids.device
        
        # Create position tensor [seq_len]
        positions = torch.arange(seq_len, device=device).float()
        
        # Compute angles: [seq_len, dim//2]
        freqs = torch.outer(positions, self.inv_freq)
        
        # Create [cos, sin, cos, sin, ...] pattern
        emb = torch.cat([freqs, freqs], dim=-1)
        emb = emb.unsqueeze(1)  # [seq_len, 1, dim]
        
        return emb


class TransformerBlock(nn.Module):
    """Single transformer block with causal attention and RoPE."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_rope=True, rope_dim=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.head_dim = d_model // n_heads
        
        # Self-attention projections (no bias terms)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.rope_dim = rope_dim if rope_dim is not None else self.head_dim
        
        self.norm1 = RMSNorm(d_model)
        
        # Feed-forward with SwiGLU
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.ff_dropout = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        """
        x: [batch_size, seq_len, d_model]
        rotary_pos_emb: [seq_len, 1, rope_dim] - RoPE frequencies
        attention_mask: [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len] 
                       - boolean mask where True means mask out (set to -inf)
                       - If None, creates causal mask automatically
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm residual connection before attention 
        residual = x
        x = self.norm1(x)
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, seq, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        if self.use_rope and rotary_pos_emb is not None:
            # Reshape for RoPE: [seq_len, batch*n_heads, head_dim]
            q_rope = q.permute(1, 0, 2, 3).contiguous().view(seq_len, batch_size * self.n_heads, self.head_dim)
            k_rope = k.permute(1, 0, 2, 3).contiguous().view(seq_len, batch_size * self.n_heads, self.head_dim)
            
            # Apply RoPE (only to first rope_dim dimensions)
            if self.rope_dim < self.head_dim:
                q_rope_part = q_rope[..., :self.rope_dim]
                k_rope_part = k_rope[..., :self.rope_dim]
                q_rope_rest = q_rope[..., self.rope_dim:]
                k_rope_rest = k_rope[..., self.rope_dim:]
                
                # Expand rotary_pos_emb to match batch*n_heads
                rope_emb = rotary_pos_emb.expand(-1, batch_size * self.n_heads, -1)
                q_rope_part = apply_rotary_pos_emb(q_rope_part, rope_emb)
                k_rope_part = apply_rotary_pos_emb(k_rope_part, rope_emb)
                
                q_rope = torch.cat([q_rope_part, q_rope_rest], dim=-1)
                k_rope = torch.cat([k_rope_part, k_rope_rest], dim=-1)
            else:
                rope_emb = rotary_pos_emb.expand(-1, batch_size * self.n_heads, -1)
                q_rope = apply_rotary_pos_emb(q_rope, rope_emb)
                k_rope = apply_rotary_pos_emb(k_rope, rope_emb)
            
            # Reshape back after RoPE: [batch, seq, n_heads, head_dim]
            q = q_rope.view(seq_len, batch_size, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k_rope.view(seq_len, batch_size, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            # Standard multi-head format: [batch, n_heads, seq, head_dim]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
        
        v = v.permute(0, 2, 1, 3)  # [batch, n_heads, seq, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Handle different mask formats
            if attention_mask.dim() == 3:
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                # [batch_size, n_heads, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
                # Expand to match number of heads if needed
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, self.n_heads, -1, -1)
                elif attention_mask.size(1) != self.n_heads:
                    raise ValueError(f"attention_mask has {attention_mask.size(1)} heads but model has {self.n_heads} heads")
            
            # Ensure mask is boolean
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            
            # Verify shape matches scores: [batch_size, n_heads, seq_len, seq_len]
            if attention_mask.shape != scores.shape:
                raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match scores shape {scores.shape}")
            
            # Mask out positions where attention_mask is True
            scores = scores.masked_fill(attention_mask, float('-inf'))
        else:
            # Create causal mask (lower triangular) if no mask provided
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            # Expand to [batch_size, n_heads, seq_len, seq_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, -1, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq, head_dim]
        
        # Reshape and project output 
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        x = residual + attn_out
        
        # Feed-forward pathway 
        residual = x
        x = self.norm2(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = swiglu(gate, up)
        ff_out = self.down_proj(activated)
        ff_out = self.ff_dropout(ff_out)
        x = residual + ff_out
        
        return x


class ARModel(nn.Module):
    """
    Autoregressive transformer model for left-to-right language modeling.
    Simplified version of GPTModel from pretrain_gpt.py
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=None,  # If None, computed via paper formula
        max_seq_len=512,
        dropout=0.1,
        eod_token_id=None,
        use_rope=True,
        rope_theta=10000.0,
        rope_percent=1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.eod_token_id = eod_token_id
        self.use_rope = use_rope
        
        # Compute FFN hidden size using h_f = floor(8 * d_model / (3 * 64)) * 64
        if d_ff is None:
            d_ff = int(math.floor(8 * d_model / (3 * 64)) * 64)

        self.d_ff = d_ff
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Position embeddings if not using RoPE
        if not use_rope:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Rotary position embeddings
        if use_rope:
            head_dim = d_model // n_heads
            rope_dim = int(head_dim * rope_percent) if rope_percent < 1.0 else head_dim
            self.rotary_pos_emb = RotaryPositionalEmbedding(rope_dim, theta=rope_theta, max_seq_len=max_seq_len)
        else:
            self.rotary_pos_emb = None
        
        # Transformer layers
        rope_dim = int((d_model // n_heads) * rope_percent) if use_rope else None
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_rope=use_rope,
                rope_dim=rope_dim
            )
            for _ in range(n_layers)
        ])
        
        # Output layer (no bias)
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, position_ids=None, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - input tokens
            position_ids: [batch_size, seq_len] - position indices (optional)
            attention_mask: [batch_size, seq_len, seq_len] - attention mask (optional, overrides causal mask)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        if self.use_rope:
            # Generate RoPE frequencies
            rotary_pos_emb = self.rotary_pos_emb(position_ids)  # [seq_len, 1, rope_dim]
            x = self.embed_dropout(token_embeds)
        else:
            pos_embeds = self.position_embedding(position_ids)
            x = self.embed_dropout(token_embeds + pos_embeds)
            rotary_pos_emb = None
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        
        # Output projection 
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

