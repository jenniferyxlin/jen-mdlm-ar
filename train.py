"""
Training script for MDLM with distributed training support
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import yaml

from model import MDLModel
from diffusion import get_masked_batch, masked_cross_entropy_loss
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


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_epoch(model, dataloader, optimizer, device, args, rank=0, use_hf_data=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Use tqdm only on rank 0
    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0)) if rank == 0 else dataloader
    
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / n_batches})
    
    return total_loss / n_batches


def evaluate(model, dataloader, device, args, rank=0, use_hf_data=False):
    """Evaluate model."""
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


def main_worker(rank, world_size, args):
    """Main training function for distributed training."""
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device(args.device)
    
    # Create model
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
    use_hf_data = args.hf_repo is not None
    use_hf_raw = args.hf_raw_repo is not None
    sampler = None  # Only used for simple text dataset with distributed training
    
    if use_hf_data:
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
    elif use_hf_raw:
        # Use raw HuggingFace dataset with proper preprocessing pipeline
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        if rank == 0:
            print(f"Loading raw dataset from HuggingFace: {args.hf_raw_repo}")
            split_name = args.hf_raw_split or args.hf_split
            print(f"  Split: {split_name}")
            print(f"  Streaming: {args.hf_streaming}")
            print(f"  Tokenizer: {args.hf_tokenizer}")
            print(f"  Text column: {args.hf_text_column}")
        
        # Set default block_size to max_seq_len if not provided
        if args.hf_block_size is None:
            args.hf_block_size = args.max_seq_len
        
        if rank == 0:
            print(f"  Block size: {args.hf_block_size}")
        
        # Load dataset (with streaming support)
        if args.hf_streaming:
            ds = load_dataset(args.hf_raw_repo, split_name, streaming=True)
            train_ds = ds["train"] if isinstance(ds, dict) else ds
        else:
            # Load dataset (only on rank 0 to avoid multiple downloads)
            if rank == 0:
                hf_dataset = load_dataset(args.hf_raw_repo, split_name)
                train_ds = hf_dataset["train"] if isinstance(hf_dataset, dict) else hf_dataset
            else:
                # Other ranks wait a bit then load
                import time
                time.sleep(1)
                hf_dataset = load_dataset(args.hf_raw_repo, split_name)
                train_ds = hf_dataset["train"] if isinstance(hf_dataset, dict) else hf_dataset
        
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
            from torch.utils.data import IterableDataset
            
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    if rank == 0:
        print(f"Starting training on {device}...")
        if world_size > 1:
            print(f"Using {world_size} GPUs with distributed training")
    
    for epoch in range(args.epochs):
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
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, dataloader, optimizer, device, args, rank, use_hf_data=use_hf_data)
        
        if rank == 0:
            print(f"Train loss: {train_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_for_info.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint to {args.save_dir}")
    
    if world_size > 1:
        cleanup_distributed()
    
    if rank == 0:
        print("Training complete!")


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
    parser = argparse.ArgumentParser(description='Train Enhanced MDM')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (e.g., default.yaml)')
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--data_file', type=str, default=None, help='Path to text data file (one sentence per line)')
    
    # HuggingFace dataset arguments
    parser.add_argument('--hf_repo', type=str, default=None, 
                        help='HuggingFace dataset repo (e.g., "org/fineweb-edu_gpt2"). If provided, uses pretokenized HF data instead of text file.')
    parser.add_argument('--hf_split', type=str, default='train', choices=['train', 'validation'],
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
    
    # Determine world size
    if args.distributed:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        world_size = args.world_size
    
    if world_size > 1 and torch.cuda.is_available():
        # Launch distributed training
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