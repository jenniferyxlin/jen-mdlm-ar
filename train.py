"""
Training script for MDLM with distributed training support
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import yaml
import json
import time
import glob
from datetime import datetime

from mdlm import MDLModel
from arm import ARModel
from diffusion import get_masked_batch, masked_cross_entropy_loss
from ar import get_ar_batch, ar_cross_entropy_loss
from data import get_dataloader


class SimpleTextDataset(Dataset):
    """Simple dataset that tokenizes text on the fly."""
    
    def __init__(self, texts, tokenizer, max_length=512, eod_token=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eod_token = eod_token
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple tokenization: split by spaces and map to integers
        tokens = text.split()[:self.max_length]
        # Convert to token IDs (simple hash-based mapping)
        token_ids = [hash(token) % (self.tokenizer.vocab_size - 1) for token in tokens]
        
        # Add EOD token if specified
        if self.eod_token is not None:
            token_ids.append(self.eod_token)
        
        # Pad or truncate to max_length + 1 (for EOD)
        target_len = self.max_length + 1 if self.eod_token is not None else self.max_length
        if len(token_ids) < target_len:
            token_ids = token_ids + [0] * (target_len - len(token_ids))
        else:
            token_ids = token_ids[:target_len]
        
        return torch.tensor(token_ids, dtype=torch.long)


class SimpleTokenizer:
    """Minimal tokenizer for demonstration."""
    
    def __init__(self, vocab_size=50257, eod_token=None):
        self.vocab_size = vocab_size
        self.eod = eod_token if eod_token is not None else vocab_size - 1


class HFRawTextDataset(Dataset):
    """Dataset for raw text from HuggingFace datasets with proper preprocessing."""
    
    def __init__(self, processed_dataset, max_length=512, eod_token=None):
        """
        Args:
            processed_dataset: Preprocessed HuggingFace dataset (filtered, tokenized, chunked)
            max_length: Maximum sequence length (should match block_size used in preprocessing)
            eod_token: End-of-document token ID (optional)
        """
        self.dataset = processed_dataset
        self.max_length = max_length
        self.eod_token = eod_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get tokenized sequence
        item = self.dataset[idx]
        
        # Handle both dict format and list format
        if isinstance(item, dict):
            token_ids = item['input_ids']
        else:
            token_ids = item
        
        # Convert to tensor
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Truncate if needed (shouldn't happen if chunking is correct, but safety check)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        return token_ids


def setup_distributed(rank, world_size, backend='nccl', master_port=None):
    """Initialize distributed training."""
    # Always respect MASTER_PORT if set in environment (from torchrun or user)
    # Only set it if not already present
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    
    # If master_port is explicitly provided, use it (overrides environment)
    # Otherwise, use environment if set, or default to 25419
    target_port = None
    if master_port:
        target_port = str(master_port)
    elif 'MASTER_PORT' in os.environ:
        env_port = os.environ['MASTER_PORT']
        # If it's 12355 (torchrun default) and we want 25419, override it
        if env_port == '12355':
            target_port = '25419'  # Override torchrun's default
        else:
            target_port = env_port  # Use what's in environment
    else:
        target_port = '25419'  # Default
    
    # Set the port 
    os.environ['MASTER_PORT'] = target_port
    
    # Double-check it's set correctly (sometimes torchrun overrides it)
    if os.environ.get('MASTER_PORT') != target_port:
        print(f"[WARNING] MASTER_PORT was overridden! Expected {target_port}, got {os.environ.get('MASTER_PORT')}")
        os.environ['MASTER_PORT'] = target_port
    
    # Debug: Print which port we're using (only on rank 0)
    if rank == 0:
        actual_port = os.environ.get('MASTER_PORT', 'NOT SET')
        actual_addr = os.environ.get('MASTER_ADDR', 'NOT SET')
        print(f"[DEBUG] Distributed setup:")
        print(f"[DEBUG]   MASTER_ADDR={actual_addr}")
        print(f"[DEBUG]   MASTER_PORT={actual_port}")
        print(f"[DEBUG]   RANK={rank}, WORLD_SIZE={world_size}")
        if actual_port == '12355':
            print(f"[WARNING] Port is 12355 - this might cause conflicts!")
    
    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def load_hf_dataset(hf_raw_repo, config_name, split_to_use, rank, streaming=False):
    """Helper function to load HuggingFace dataset with proper rank handling."""
    from datasets import load_dataset
    
    if streaming:
        if config_name:
            return load_dataset(hf_raw_repo, config_name, streaming=True, split=split_to_use)
        else:
            return load_dataset(hf_raw_repo, streaming=True, split=split_to_use)
    else:
        # Load dataset (only on rank 0 to avoid multiple downloads)
        if rank == 0:
            if config_name:
                hf_dataset = load_dataset(hf_raw_repo, config_name)
            else:
                hf_dataset = load_dataset(hf_raw_repo)
            train_ds = hf_dataset[split_to_use] if isinstance(hf_dataset, dict) and split_to_use in hf_dataset else hf_dataset
        else:
            # Other ranks wait a bit then load
            time.sleep(1)
            if config_name:
                hf_dataset = load_dataset(hf_raw_repo, config_name)
            else:
                hf_dataset = load_dataset(hf_raw_repo)
            train_ds = hf_dataset[split_to_use] if isinstance(hf_dataset, dict) and split_to_use in hf_dataset else hf_dataset
        return train_ds


def train_epoch_mdlm(model, dataloader, optimizer, device, args, rank=0, use_hf_data=False, metrics_logger=None, epoch_num=0):
    """Train MDLM for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    step_losses = []  # Track losses per step for visualization
    running_avg_window = 100  # Window for running average
    recent_losses = []  # Track recent losses for running average
    
    # Use tqdm only on rank 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num+1}", disable=(rank != 0)) if rank == 0 else dataloader
    
    for step, batch in enumerate(pbar):
        # Handle HuggingFace data format (x, y) tuples vs simple token tensors
        if use_hf_data:
            x, y = batch
            # Reconstruct full sequence: x is seq[:-1], y is seq[1:], so full seq = [x, last_token_of_y]
            tokens = torch.cat([x, y[:, -1:]], dim=1).to(device)
        else:
            tokens = batch.to(device)
        
        # Apply masked diffusion forward process
        noisy_input, labels, EOD_mask, masked_indices, p_mask, position_ids = get_masked_batch(
            tokens,
            eps=args.eps,
            mask_token_id=model.module.mask_token_id if isinstance(model, DDP) else model.mask_token_id,
            device=device,
            eod_token=args.eod_token,
            mask_schedule=args.mask_schedule
        )
        
        # Forward pass
        logits = model(noisy_input, position_ids=position_ids)
        
        # Compute loss
        loss = masked_cross_entropy_loss(logits, labels, EOD_mask, masked_indices, p_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        
        loss_value = loss.item()
        total_loss += loss_value
        n_batches += 1
        step_losses.append(loss_value)
        
        # Track running average
        recent_losses.append(loss_value)
        if len(recent_losses) > running_avg_window:
            recent_losses.pop(0)
        running_avg = sum(recent_losses) / len(recent_losses) if recent_losses else loss_value
        
        # Log step-level metrics
        if metrics_logger is not None and rank == 0:
            metrics_logger.log_step(step, loss_value, total_loss / n_batches)
        
        # Update progress bar with enhanced loss info
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg': f'{total_loss / n_batches:.4f}',
                'run_avg': f'{running_avg:.4f}'
            })
    
    avg_loss = total_loss / n_batches
    return avg_loss, step_losses


def train_epoch_ar(model, dataloader, optimizer, device, args, rank=0, use_hf_data=False, metrics_logger=None, epoch_num=0):
    """Train AR model for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    step_losses = []  # Track losses per step for visualization
    running_avg_window = 100  # Window for running average
    recent_losses = []  # Track recent losses for running average
    
    # Use tqdm only on rank 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num+1}", disable=(rank != 0)) if rank == 0 else dataloader
    
    for step, batch in enumerate(pbar):
        # Handle HuggingFace data format (x, y) tuples vs simple token tensors
        if use_hf_data:
            x, y = batch
            # For AR: x is input, y is target (already shifted)
            # We need full sequence: [x, last_token_of_y] to create input/target pairs
            tokens = torch.cat([x, y[:, -1:]], dim=1).to(device)
        else:
            tokens = batch.to(device)
        
        # Prepare AR batch (shifts tokens and creates masks)
        randmask_ratio = getattr(args, 'randmask_ratio', 0.0)
        eps = getattr(args, 'eps', 1e-3)
        input_ids, labels, loss_mask, attention_mask, position_ids = get_ar_batch(
            tokens,
            eod_token=args.eod_token,
            eod_mask_loss=True,
            randmask_ratio=randmask_ratio,
            eps=eps
        )
        
        # Forward pass
        logits = model(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = ar_cross_entropy_loss(logits, labels, loss_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        
        loss_value = loss.item()
        total_loss += loss_value
        n_batches += 1
        step_losses.append(loss_value)
        
        # Track running average
        recent_losses.append(loss_value)
        if len(recent_losses) > running_avg_window:
            recent_losses.pop(0)
        running_avg = sum(recent_losses) / len(recent_losses) if recent_losses else loss_value
        
        # Log step-level metrics
        if metrics_logger is not None and rank == 0:
            metrics_logger.log_step(step, loss_value, total_loss / n_batches)
        
        # Update progress bar with enhanced loss info
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg': f'{total_loss / n_batches:.4f}',
                'run_avg': f'{running_avg:.4f}'
            })
    
    avg_loss = total_loss / n_batches
    return avg_loss, step_losses


def evaluate_mdlm(model, dataloader, device, args, rank=0, use_hf_data=False):
    """Evaluate MDLM model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", disable=(rank != 0)) if rank == 0 else dataloader
        for batch in pbar:
            # Handle HuggingFace data format (x, y) tuples vs simple token tensors
            if use_hf_data:
                x, y = batch
                # Reconstruct full sequence: x is seq[:-1], y is seq[1:], so full seq = [x, last_token_of_y]
                tokens = torch.cat([x, y[:, -1:]], dim=1).to(device)
            else:
                tokens = batch.to(device)
            
            # Apply masked diffusion forward process
            noisy_input, labels, EOD_mask, masked_indices, p_mask, position_ids = get_masked_batch(
                tokens,
                eps=args.eps,
                mask_token_id=model.module.mask_token_id if isinstance(model, DDP) else model.mask_token_id,
                device=device,
                eod_token=args.eod_token,
                mask_schedule=args.mask_schedule
            )
            
            # Forward pass
            logits = model(noisy_input, position_ids=position_ids)
            
            # Compute loss
            loss = masked_cross_entropy_loss(logits, labels, EOD_mask, masked_indices, p_mask)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def evaluate_ar(model, dataloader, device, args, rank=0, use_hf_data=False):
    """Evaluate AR model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", disable=(rank != 0)) if rank == 0 else dataloader
        for batch in pbar:
            # Handle HuggingFace data format (x, y) tuples vs simple token tensors
            if use_hf_data:
                x, y = batch
                # For AR: x is input, y is target (already shifted)
                tokens = torch.cat([x, y[:, -1:]], dim=1).to(device)
            else:
                tokens = batch.to(device)
            
            # Prepare AR batch (no random masking during evaluation)
            input_ids, labels, loss_mask, attention_mask, position_ids = get_ar_batch(
                tokens,
                eod_token=args.eod_token,
                eod_mask_loss=True,
                randmask_ratio=0.0  # No random masking during evaluation
            )
            
            # Forward pass
            logits = model(input_ids, position_ids=position_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = ar_cross_entropy_loss(logits, labels, loss_mask)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


class MetricsLogger:
    """Logger for training metrics to enable visualization and comparison."""
    
    def __init__(self, save_dir, model_type="MDLM", experiment_name=None):
        self.save_dir = save_dir
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = {
            'model_type': model_type,
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'steps': [],
            'step_losses': [],
            'epoch_losses': [],
            'epoch_avg_losses': [],
            'validation_losses': []
        }
        self.current_epoch = 0
        self.global_step = 0
        
    def log_step(self, step, loss, avg_loss):
        """Log metrics for a single training step."""
        self.global_step += 1
        self.metrics['steps'].append(self.global_step)
        self.metrics['step_losses'].append(loss)
        
    def log_epoch(self, epoch, epoch_loss, step_losses=None, validation_loss=None):
        """Log metrics for a completed epoch."""
        self.current_epoch = epoch
        self.metrics['epochs'].append(epoch + 1)
        self.metrics['epoch_losses'].append(epoch_loss)
        if step_losses:
            self.metrics['epoch_avg_losses'].append(sum(step_losses) / len(step_losses))
        if validation_loss is not None:
            self.metrics['validation_losses'].append(validation_loss)
        
    def save(self):
        """Save metrics to JSON file."""
        self.metrics['end_time'] = datetime.now().isoformat()
        metrics_file = os.path.join(self.save_dir, f'{self.experiment_name}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return metrics_file


def main_worker(rank, world_size, args):
    """Main training function for distributed training."""
    # If using torchrun, get rank and world_size from environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size, master_port=args.master_port)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device(args.device)
    
    # Determine model type
    model_type = getattr(args, 'model_type', 'mdlm').lower()
    if model_type not in ['mdlm', 'ar']:
        if rank == 0:
            print(f"Warning: Unknown model_type '{model_type}', defaulting to 'mdlm'")
        model_type = 'mdlm'
    
    # Create model based on type
    if model_type == 'ar':
        model = ARModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            eod_token_id=args.eod_token,
            use_rope=args.use_rope,
            rope_theta=args.rope_theta,
            rope_percent=args.rope_percent
        ).to(device)
    else:  # mdlm
        model = MDLModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_rope=args.use_rope,
            rope_theta=args.rope_theta,
            rope_percent=args.rope_percent
        ).to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        model_for_info = model.module
    else:
        model_for_info = model
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using RoPE: {args.use_rope}")
    
    # Determine data source and create dataloader
    # Priority: hf_raw_repo > hf_repo > simple text dataset
    # (hf_raw_repo takes precedence if both are set)
    use_hf_raw = args.hf_raw_repo is not None
    use_hf_data = args.hf_repo is not None and not use_hf_raw  # Only use if hf_raw_repo is not set
    sampler = None  # Only used for simple text dataset with distributed training
    
    if use_hf_raw:
        # Use raw HuggingFace dataset with proper preprocessing pipeline
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # hf_raw_split can be a config name (e.g., "wikitext-2-v1") or None
        config_name = args.hf_raw_split
        # hf_split is the actual split to use (train, test, validation)
        split_to_use = args.hf_split
        
        if rank == 0:
            print(f"Loading raw dataset from HuggingFace: {args.hf_raw_repo}")
            if config_name:
                print(f"  Config: {config_name}")
            print(f"  Split: {split_to_use}")
            print(f"  Streaming: {args.hf_streaming}")
            print(f"  Tokenizer: {args.hf_tokenizer}")
            print(f"  Text column: {args.hf_text_column}")
        
        # Set default block_size to max_seq_len if not provided
        if args.hf_block_size is None:
            args.hf_block_size = args.max_seq_len
        
        if rank == 0:
            print(f"  Block size: {args.hf_block_size}")
        
        # Load dataset (with streaming support)
        train_ds = load_hf_dataset(
            args.hf_raw_repo, 
            config_name, 
            split_to_use, 
            rank, 
            streaming=args.hf_streaming
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Update vocab_size from tokenizer if needed
        if args.vocab_size == 50257:  # Default value
            args.vocab_size = len(tokenizer)
            if rank == 0:
                print(f"  Using vocab_size from tokenizer: {args.vocab_size}")
        
        # Set EOD token from tokenizer
        if args.eod_token is None or args.eod_token == args.vocab_size - 1:
            args.eod_token = tokenizer.eos_token_id or args.vocab_size - 1
        
        # Preprocessing pipeline
        if rank == 0:
            print("  Preprocessing dataset: filtering empty rows...")
        
        # Step 1: Filter empty/newline-only rows
        filtered_ds = train_ds.filter(
            lambda x: bool(x[args.hf_text_column] and x[args.hf_text_column].strip())
        )
        
        if rank == 0:
            print("  Tokenizing dataset (batched)...")
        
        # Step 2: Tokenize in batches
        def tokenize(batch):
            """Tokenize text batches."""
            return tokenizer(batch[args.hf_text_column], return_attention_mask=False)
        
        tokenized = filtered_ds.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in filtered_ds.column_names if col != "input_ids"]
        )
        
        if rank == 0:
            print(f"  Chunking sequences into blocks of size {args.hf_block_size}...")
        
        # Step 3: Chunk sequences into fixed-size blocks
        block_size = args.hf_block_size
        
        def chunk(batch):
            """Concatenate and chunk sequences."""
            concatenated = sum(batch["input_ids"], [])
            total_len = (len(concatenated) // block_size) * block_size
            return {
                "input_ids": [
                    concatenated[i : i + block_size]
                    for i in range(0, total_len, block_size)
                ]
            }
        
        processed_ds = tokenized.map(chunk, batched=True)
        
        # For streaming datasets, we need to use IterableDataset
        if args.hf_streaming:
            # Convert to IterableDataset for streaming
            # Note: streaming datasets can't use DistributedSampler easily
            class StreamingDatasetWrapper(IterableDataset):
                def __init__(self, stream_ds, rank, world_size):
                    self.stream_ds = stream_ds
                    self.rank = rank
                    self.world_size = world_size
                
                def __iter__(self):
                    for i, item in enumerate(self.stream_ds):
                        # Simple round-robin for distributed
                        if i % self.world_size == self.rank:
                            yield torch.tensor(item["input_ids"], dtype=torch.long)
            
            dataset = StreamingDatasetWrapper(processed_ds, rank, world_size)
            # For streaming, we can't use DistributedSampler
            sampler = None
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers or 0
            )
        else:
            # Create dataset wrapper
            dataset = HFRawTextDataset(
                processed_ds,
                max_length=args.hf_block_size,
                eod_token=args.eod_token
            )
            
            # Create distributed sampler if using multiple GPUs
            if world_size > 1:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers or 0)
            else:
                sampler = None
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers or 0)
    elif use_hf_data:
        # Use HuggingFace pretokenized dataset
        if rank == 0:
            print(f"Loading pretokenized dataset from HuggingFace: {args.hf_repo}")
            print(f"  Split: {args.hf_split}")
            print(f"  Max tokens: {args.hf_max_tokens if args.hf_max_tokens else 'all'}")
            print(f"  Version: {args.hf_version if args.hf_version else 'latest'}")
        
        dataloader = get_dataloader(
            hf_repo=args.hf_repo,
            batch_size=args.batch_size,
            seq_len=args.max_seq_len,
            rank=rank,
            world_size=world_size,
            split=args.hf_split,
            max_tokens=args.hf_max_tokens,
            epoch=0,  # Will be updated in training loop
            version=args.hf_version,
            subset=args.hf_subset,
            replicate_shards=args.hf_replicate_shards,
            num_workers=args.num_workers,
        )
    else:
        # Use simple text dataset
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size, eod_token=args.eod_token)
        
        # Create dummy data if no data file provided
        if args.data_file is None:
            if rank == 0:
                print("No data file provided, using dummy data for demonstration")
            dummy_texts = [
                "This is a sample sentence for training the model.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
            ] * 100  # Repeat to have enough data
        else:
            if rank == 0:
                print(f"Loading data from file: {args.data_file}")
            with open(args.data_file, 'r') as f:
                dummy_texts = [line.strip() for line in f if line.strip()]
        
        dataset = SimpleTextDataset(
            dummy_texts, tokenizer, max_length=args.max_seq_len, eod_token=args.eod_token
        )
        
        # Create distributed sampler if using multiple GPUs
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            sampler = None
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create validation dataloader if using HuggingFace dataset
    val_dataloader = None
    if (use_hf_raw or use_hf_data) and rank == 0:
        # Try to load validation split
        if use_hf_raw:
            try:
                if rank == 0:
                    print("  Loading validation dataset...")
                val_ds = load_hf_dataset(
                    args.hf_raw_repo,
                    config_name,
                    'validation',  # Use validation split
                    rank,
                    streaming=False  # Don't stream validation
                )
                # Apply same preprocessing as training data
                filtered_val_ds = val_ds.filter(
                    lambda x: bool(x[args.hf_text_column] and x[args.hf_text_column].strip())
                )
                
                tokenized_val = filtered_val_ds.map(
                    lambda batch: tokenizer(batch[args.hf_text_column], return_attention_mask=False),
                    batched=True,
                    remove_columns=[col for col in filtered_val_ds.column_names if col != "input_ids"]
                )
                
                processed_val_ds = tokenized_val.map(
                    lambda batch: {
                        "input_ids": [
                            sum(batch["input_ids"], [])[i : i + args.hf_block_size]
                            for i in range(0, (len(sum(batch["input_ids"], [])) // args.hf_block_size) * args.hf_block_size, args.hf_block_size)
                        ]
                    },
                    batched=True
                )
                
                val_dataset = HFRawTextDataset(
                    processed_val_ds,
                    max_length=args.hf_block_size,
                    eod_token=args.eod_token
                )
                if world_size > 1:
                    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0)
                else:
                    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                if rank == 0:
                    print(f"  Validation dataset loaded: {len(val_dataset)} samples")
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not load validation set: {e}")
        elif use_hf_data:
            try:
                val_dataloader = get_dataloader(
                    hf_repo=args.hf_repo,
                    batch_size=args.batch_size,
                    seq_len=args.max_seq_len,
                    rank=rank,
                    world_size=world_size,
                    split='validation',
                    max_tokens=None,  # Use all validation data
                    epoch=0,
                    version=args.hf_version,
                    subset=args.hf_subset,
                    replicate_shards=True,  # All ranks see all validation data
                    num_workers=0,  # No workers for validation
                )
                if rank == 0:
                    print("  Validation dataloader created")
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not create validation dataloader: {e}")
    
    # Broadcast val_dataloader availability to all ranks (simplified: just check if it exists on rank 0)
    has_val = val_dataloader is not None
    if world_size > 1:
        has_val_tensor = torch.tensor([1 if has_val else 0], dtype=torch.long, device=device)
        dist.broadcast(has_val_tensor, src=0)
        has_val = bool(has_val_tensor.item())
        # Other ranks need to create their own validation dataloader if rank 0 has one
        if has_val and rank > 0:
            if use_hf_raw:
                try:
                    val_ds = load_hf_dataset(args.hf_raw_repo, config_name, 'validation', rank, streaming=False)
                    filtered_val_ds = val_ds.filter(lambda x: bool(x[args.hf_text_column] and x[args.hf_text_column].strip()))
                    tokenized_val = filtered_val_ds.map(
                        lambda batch: tokenizer(batch[args.hf_text_column], return_attention_mask=False),
                        batched=True, remove_columns=[col for col in filtered_val_ds.column_names if col != "input_ids"]
                    )
                    processed_val_ds = tokenized_val.map(
                        lambda batch: {
                            "input_ids": [
                                sum(batch["input_ids"], [])[i : i + args.hf_block_size]
                                for i in range(0, (len(sum(batch["input_ids"], [])) // args.hf_block_size) * args.hf_block_size, args.hf_block_size)
                            ]
                        }, batched=True
                    )
                    val_dataset = HFRawTextDataset(processed_val_ds, max_length=args.hf_block_size, eod_token=args.eod_token)
                    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0)
                except:
                    val_dataloader = None
            elif use_hf_data:
                try:
                    val_dataloader = get_dataloader(
                        hf_repo=args.hf_repo, batch_size=args.batch_size, seq_len=args.max_seq_len,
                        rank=rank, world_size=world_size, split='validation', max_tokens=None,
                        epoch=0, version=args.hf_version, subset=args.hf_subset,
                        replicate_shards=True, num_workers=0
                    )
                except:
                    val_dataloader = None
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize metrics logger (only on rank 0)
    metrics_logger = None
    start_epoch = 0
    if rank == 0:
        model_type_str = model_type.upper()
        experiment_name = args.experiment_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metrics_logger = MetricsLogger(args.save_dir, model_type=model_type_str, experiment_name=experiment_name)
        
        # Check for existing checkpoints to resume from (for this specific experiment)
        checkpoint_dir = args.save_dir
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
            if checkpoint_files:
                # Extract epoch numbers and find the latest
                epoch_numbers = []
                for f in checkpoint_files:
                    try:
                        epoch_num = int(f.replace('checkpoint_epoch_', '').replace('.pt', ''))
                        # Only consider checkpoints that are less than target epochs
                        if epoch_num < args.epochs:
                            epoch_numbers.append(epoch_num)
                    except ValueError:
                        continue
                
                if epoch_numbers:
                    # Try checkpoints from newest to oldest until we find a valid one
                    epoch_numbers.sort(reverse=True)
                    checkpoint_loaded = False
                    
                    for epoch_num in epoch_numbers:
                        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch_num}.pt')
                        
                        if not os.path.exists(checkpoint_path):
                            continue
                        
                        try:
                            # Try to load checkpoint
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                            
                            # Verify checkpoint has required keys
                            if 'epoch' not in checkpoint or 'model_state_dict' not in checkpoint:
                                print(f"Warning: Checkpoint {checkpoint_path} is missing required keys, skipping...")
                                continue
                            
                            print(f"\n{'='*60}")
                            print(f"Found valid checkpoint: {checkpoint_path}")
                            print(f"Resuming from epoch {epoch_num + 1}/{args.epochs}")
                            print(f"{'='*60}\n")
                            
                            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                            
                            # Load model state
                            if isinstance(model, DDP):
                                model.module.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            
                            # Load optimizer state if available
                            if 'optimizer_state_dict' in checkpoint:
                                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            
                            # Load metrics if available
                            metrics_file = os.path.join(checkpoint_dir, f'{experiment_name}_metrics.json')
                            if os.path.exists(metrics_file):
                                try:
                                    with open(metrics_file, 'r') as f:
                                        saved_metrics = json.load(f)
                                        metrics_logger.metrics = saved_metrics
                                        metrics_logger.current_epoch = start_epoch - 1
                                        metrics_logger.global_step = len(saved_metrics.get('steps', []))
                                        print(f"Loaded metrics from: {metrics_file}")
                                except Exception as e:
                                    print(f"Warning: Could not load metrics from {metrics_file}: {e}")
                            
                            print(f"Resuming training from epoch {start_epoch + 1}/{args.epochs}\n")
                            checkpoint_loaded = True
                            break
                            
                        except Exception as e:
                            print(f"Warning: Checkpoint {checkpoint_path} is corrupted or invalid: {e}")
                            print(f"Trying next checkpoint...")
                            continue
                    
                    if not checkpoint_loaded:
                        print("Warning: No valid checkpoints found, starting from scratch")
    
    # Broadcast start_epoch to all ranks
    if world_size > 1:
        start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.long, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = int(start_epoch_tensor.item())
    
    # Training loop
    if rank == 0:
        print(f"Starting training on {device}...")
        if world_size > 1:
            print(f"Using {world_size} GPUs with distributed training")
        if metrics_logger:
            print(f"Metrics will be saved to: {metrics_logger.save_dir}")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}/{args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        # Update epoch for HuggingFace dataloader (for shard shuffling)
        if use_hf_data:
            dataloader = get_dataloader(
                hf_repo=args.hf_repo,
                batch_size=args.batch_size,
                seq_len=args.max_seq_len,
                rank=rank,
                world_size=world_size,
                split=args.hf_split,
                max_tokens=args.hf_max_tokens,
                epoch=epoch,
                version=args.hf_version,
                subset=args.hf_subset,
                replicate_shards=args.hf_replicate_shards,
                num_workers=args.num_workers,
            )
        elif sampler is not None:
            sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")
        
        # Use appropriate training function based on model type
        if model_type == 'ar':
            train_loss, step_losses = train_epoch_ar(model, dataloader, optimizer, device, args, rank, use_hf_data=use_hf_data, metrics_logger=metrics_logger, epoch_num=epoch)
        else:
            train_loss, step_losses = train_epoch_mdlm(model, dataloader, optimizer, device, args, rank, use_hf_data=use_hf_data, metrics_logger=metrics_logger, epoch_num=epoch)
        
        if rank == 0:
            # Calculate additional statistics
            min_loss = min(step_losses) if step_losses else train_loss
            max_loss = max(step_losses) if step_losses else train_loss
            std_loss = (sum((x - train_loss)**2 for x in step_losses) / len(step_losses))**0.5 if step_losses else 0.0
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {train_loss:.4f}")
            print(f"  Min Loss:     {min_loss:.4f}")
            print(f"  Max Loss:     {max_loss:.4f}")
            print(f"  Std Dev:      {std_loss:.4f}")
            print(f"  Steps:        {len(step_losses)}")
            
            # Show improvement if not first epoch
            if epoch > 0 and metrics_logger and len(metrics_logger.metrics['epoch_losses']) > 1:
                prev_loss = metrics_logger.metrics['epoch_losses'][-2]
                improvement = prev_loss - train_loss
                improvement_pct = (improvement / prev_loss * 100) if prev_loss > 0 else 0
                print(f"  Improvement:  {improvement:+.4f} ({improvement_pct:+.2f}%)")
            
            # Run validation evaluation if validation dataloader is available
            validation_loss = None
            if val_dataloader is not None:
                if rank == 0:
                    print(f"\n  Running validation evaluation...")
                if model_type == 'ar':
                    validation_loss = evaluate_ar(model, val_dataloader, device, args, rank, use_hf_data=use_hf_data)
                else:
                    validation_loss = evaluate_mdlm(model, val_dataloader, device, args, rank, use_hf_data=use_hf_data)
                
                # Aggregate validation loss across all ranks
                if world_size > 1:
                    validation_loss_tensor = torch.tensor(validation_loss, device=device)
                    dist.all_reduce(validation_loss_tensor, op=dist.ReduceOp.SUM)
                    validation_loss = validation_loss_tensor.item() / world_size
                
                if rank == 0:
                    print(f"  Validation Loss: {validation_loss:.4f}")
            
            # Log epoch metrics
            if metrics_logger:
                metrics_logger.log_epoch(epoch, train_loss, step_losses, validation_loss=validation_loss)
            
            # Save checkpoint (with error handling for disk space issues)
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_for_info.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                }
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"\n✓ Checkpoint saved to {checkpoint_path}")
                
                # Delete old checkpoints, keeping the last 10 most recent ones
                if rank == 0:
                    checkpoint_pattern = os.path.join(args.save_dir, 'checkpoint_epoch_*.pt')
                    all_checkpoints = glob.glob(checkpoint_pattern)
                    keep_last_n = 10  # Keep the last 10 checkpoints
                    if len(all_checkpoints) > keep_last_n:
                        # Sort by epoch number (extract from filename)
                        def get_epoch_num(path):
                            basename = os.path.basename(path)
                            try:
                                return int(basename.replace('checkpoint_epoch_', '').replace('.pt', ''))
                            except:
                                return 0
                        
                        all_checkpoints.sort(key=get_epoch_num)
                        # Keep only the most recent N (last N in sorted list)
                        checkpoints_to_delete = all_checkpoints[:-keep_last_n]
                        for old_checkpoint in checkpoints_to_delete:
                            try:
                                os.remove(old_checkpoint)
                                print(f"  Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
                            except Exception as e:
                                print(f"  Warning: Could not delete {old_checkpoint}: {e}")
            except RuntimeError as e:
                if "file write failed" in str(e) or "disk" in str(e).lower() or "space" in str(e).lower():
                    print(f"\n⚠ Warning: Failed to save checkpoint (likely disk space issue): {e}")
                    print("Training will continue, but checkpoint was not saved.")
                    print("Consider downloading/moving old checkpoints to free space:")
                    print("  # Find large checkpoints: du -h ./checkpoints/*.pt | sort -rh")
                    print("  # Download old ones: scp user@server:~/jen-mdlm-ar/checkpoints/checkpoint_epoch_*.pt ~/Downloads/")
                    print("  # Then delete from server if needed: rm ./checkpoints/checkpoint_epoch_{1..50}.pt")
                else:
                    print(f"\n⚠ Warning: Failed to save checkpoint: {e}")
                    print("Training will continue, but checkpoint was not saved.")
            except Exception as e:
                print(f"\n⚠ Warning: Failed to save checkpoint: {e}")
                print("Training will continue, but checkpoint was not saved.")
    
    if world_size > 1:
        cleanup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        
        if metrics_logger:
            metrics_file = metrics_logger.save()
            print(f"✓ Training metrics saved to: {metrics_file}")
            
            # Print final summary
            if metrics_logger.metrics['epoch_losses']:
                initial_loss = metrics_logger.metrics['epoch_losses'][0]
                final_loss = metrics_logger.metrics['epoch_losses'][-1]
                total_improvement = initial_loss - final_loss
                improvement_pct = (total_improvement / initial_loss * 100) if initial_loss > 0 else 0
                
                print(f"\nFinal Training Summary:")
                print(f"  Total Epochs:     {len(metrics_logger.metrics['epoch_losses'])}")
                print(f"  Total Steps:      {len(metrics_logger.metrics['steps'])}")
                print(f"  Initial Loss:     {initial_loss:.4f}")
                print(f"  Final Loss:       {final_loss:.4f}")
                print(f"  Total Improvement: {total_improvement:+.4f} ({improvement_pct:+.2f}%)")
            
            # Automatically generate visualization for individual run
            print(f"\n{'='*60}")
            print("Generating training visualizations...")
            print(f"{'='*60}")
            try:
                import subprocess
                import sys
                # Create unique output directory for this run's plots
                # Use experiment name to create a subdirectory
                plot_output_dir = os.path.join(args.save_dir, 'plots', metrics_logger.experiment_name)
                os.makedirs(plot_output_dir, exist_ok=True)
                # Don't use --compare_epochs for individual runs (only for combined comparison)
                result = subprocess.run(
                    [sys.executable, 'visualize_training.py', metrics_file, '--output_dir', plot_output_dir],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ Visualizations saved to: {plot_output_dir}")
                    print("\nGenerated plots:")
                    print(f"  - loss_per_step.png")
                    print(f"  - loss_per_epoch.png")
                    print(f"  - loss_smoothed.png")
                else:
                    print(f"Warning: Visualization failed. You can run manually:")
                    print(f"  python visualize_training.py {metrics_file} --output_dir {plot_output_dir}")
                    if result.stderr:
                        print(f"  Error: {result.stderr}")
            except Exception as e:
                print(f"Warning: Could not generate visualizations automatically: {e}")
                print(f"You can generate them manually with:")
                print(f"  python visualize_training.py {metrics_file} --output_dir {os.path.join(args.save_dir, 'plots', metrics_logger.experiment_name)}")
        
        print(f"\n{'='*60}")


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_yaml_config(args, yaml_config):
    """Apply YAML configuration to args, with CLI args taking precedence."""
    # Map YAML keys to argument names and their defaults
    yaml_to_arg = {
        'hf_repo': ('hf_repo', None),
        'max_tokens': ('hf_max_tokens', None),
        'subset': ('hf_subset', None),
        'version': ('hf_version', None),
    }
    
    # Apply YAML config values
    for yaml_key, (arg_key, default) in yaml_to_arg.items():
        if yaml_key in yaml_config:
            # Convert null/None strings to None
            value = yaml_config[yaml_key]
            if value == 'null' or (isinstance(value, str) and value.lower() == 'null'):
                value = None
            
            # Only apply if current value is the default (meaning CLI didn't override it)
            current_value = getattr(args, arg_key, default)
            if current_value == default:
                setattr(args, arg_key, value)
    
    # Handle tokenizer -> vocab_size mapping (common tokenizers)
    if 'tokenizer' in yaml_config and yaml_config['tokenizer']:
        tokenizer_name = yaml_config['tokenizer'].lower()
        # Only set vocab_size if it's still at default
        if args.vocab_size == 50257:  # Default value
            if 'gpt2' in tokenizer_name:
                args.vocab_size = 50257
            elif 'gpt-neo' in tokenizer_name:
                args.vocab_size = 50257
            # Add more tokenizer mappings as needed
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train MDLM or AR model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (e.g., default.yaml)')
    parser.add_argument('--model_type', type=str, default='mdlm', choices=['mdlm', 'ar'],
                        help='Model type: "mdlm" for Masked Diffusion Language Model or "ar" for Autoregressive (default: mdlm)')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=None, help='Feed-forward dimension (None = use paper formula)')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--eps', type=float, default=1e-3, help='Minimum mask probability')
    parser.add_argument('--mask_schedule', type=str, default='linear', choices=['linear', 'cosine'], 
                        help='Masking schedule: linear or cosine (default: linear)')
    parser.add_argument('--randmask_ratio', type=float, default=0.0,
                        help='Probability of applying random masking to attention during AR training (0.0 = no random masking, similar to Megatron)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for metrics logging (default: auto-generated)')
    parser.add_argument('--data_file', type=str, default=None, help='Path to text data file (one sentence per line)')
    
    # HuggingFace dataset arguments
    parser.add_argument('--hf_repo', type=str, default=None, 
                        help='HuggingFace dataset repo (e.g., "org/fineweb-edu_gpt2"). If provided, uses pretokenized HF data instead of text file.')
    parser.add_argument('--hf_split', type=str, default='train', choices=['train', 'validation', 'test'],
                        help='Dataset split to use (default: train)')
    parser.add_argument('--hf_max_tokens', type=int, default=None,
                        help='Maximum tokens to use from dataset (deterministic subset). None = use all.')
    parser.add_argument('--hf_version', type=str, default=None,
                        help='Dataset version tag (e.g., "v1"). None = latest.')
    parser.add_argument('--hf_subset', type=str, default=None,
                        help='Subdirectory in the repo to load from.')
    parser.add_argument('--hf_replicate_shards', action='store_true',
                        help='If True, all ranks load all shards but filter sequences. Useful for validation.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of DataLoader workers. Default: 4 for training, 0 for validation.')
    
    # Raw HuggingFace dataset arguments (for non-pretokenized datasets)
    parser.add_argument('--hf_raw_repo', type=str, default=None,
                        help='Raw HuggingFace dataset repo (e.g., "Salesforce/wikitext"). If provided, loads raw text and tokenizes on-the-fly.')
    parser.add_argument('--hf_raw_split', type=str, default=None,
                        help='Split name for raw HF dataset (e.g., "wikitext-103-v1"). If None, uses hf_split.')
    parser.add_argument('--hf_tokenizer', type=str, default='gpt2',
                        help='Tokenizer name from transformers library (default: gpt2)')
    parser.add_argument('--hf_text_column', type=str, default='text',
                        help='Column name containing text in the dataset (default: text)')
    parser.add_argument('--hf_block_size', type=int, default=None,
                        help='Block size for chunking sequences (default: max_seq_len)')
    parser.add_argument('--hf_streaming', action='store_true',
                        help='Use streaming mode for large datasets (loads data on-the-fly)')
    
    # RoPE arguments
    parser.add_argument('--use_rope', action='store_true', default=True, help='Use Rotary Position Embeddings')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--rope_percent', type=float, default=1.0, help='Percentage of dimensions to use for RoPE')
    
    # EOD token
    parser.add_argument('--eod_token', type=int, default=None, help='End-of-document token ID')
    
    # Distributed training
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs for distributed training')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--master_port', type=str, default=None, help='Master port for distributed training (default: 25419, or from env/--master-port flag)')
    
    args = parser.parse_args()
    
    # Load YAML config if provided or if default.yaml exists
    if args.config:
        if not os.path.exists(args.config):
            print(f"Warning: Config file {args.config} not found. Continuing with CLI args only.")
        else:
            yaml_config = load_yaml_config(args.config)
            args = apply_yaml_config(args, yaml_config)
            print(f"Loaded configuration from {args.config}")
    elif os.path.exists('default.yaml'):
        # Auto-load default.yaml if it exists and no config specified
        yaml_config = load_yaml_config('default.yaml')
        args = apply_yaml_config(args, yaml_config)
        print("Auto-loaded configuration from default.yaml")
    
    # Set EOD token if not specified
    if args.eod_token is None:
        args.eod_token = args.vocab_size - 1
    
    # Check if running under torchrun (which sets RANK and WORLD_SIZE)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun is managing the processes - just call main_worker directly
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        main_worker(rank, world_size, args)
    else:
        # Determine world size for manual spawn
        if args.distributed:
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            world_size = args.world_size
        
        if world_size > 1 and torch.cuda.is_available():
            # Launch distributed training using spawn
            torch.multiprocessing.spawn(
                main_worker,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        else:
            # Single GPU/CPU training
            main_worker(0, 1, args)


if __name__ == '__main__':
    main()