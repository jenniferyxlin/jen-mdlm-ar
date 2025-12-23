# Dataset Usage Guide

This guide explains how to choose and use datasets with the MDLM training script.

## Option 1: HuggingFace Pretokenized Dataset (Recommended)

The training script supports pretokenized datasets from HuggingFace Hub that are stored as sharded Parquet files.

### Basic Usage

```bash
python train.py --hf_repo "org/dataset-name" --vocab_size 50257
```

### Example: FineWeb Dataset

```bash
python train.py \
    --hf_repo "HuggingFaceFW/fineweb" \
    --hf_split train \
    --vocab_size 50257 \
    --max_seq_len 512 \
    --batch_size 32 \
    --epochs 10
```

### Dataset Requirements

The dataset must be:
- **Pretokenized**: Tokens should already be converted to integer IDs
- **Sharded**: Stored as multiple `.parquet` files (shards)
- **Metadata**: Include a `metadata.json` file with `shard_size` and `chunk_size`
- **Format**: Each shard should have a `tokens` column with token arrays

### Available Options

| Argument | Description | Example |
|----------|-------------|---------|
| `--hf_repo` | HuggingFace dataset repository | `"HuggingFaceFW/fineweb"` |
| `--hf_split` | Dataset split to use | `train` or `validation` |
| `--hf_max_tokens` | Limit tokens (for testing) | `1000000` (1M tokens) |
| `--hf_version` | Dataset version tag | `"v1"` or `None` for latest |
| `--hf_subset` | Subdirectory in repo | `"subset_name"` |
| `--hf_replicate_shards` | Replicate shards for validation | Flag (no value) |
| `--num_workers` | DataLoader workers | `4` (default: 4 for train, 0 for val) |

### Complete Example with All Options

```bash
python train.py \
    --hf_repo "HuggingFaceFW/fineweb" \
    --hf_split train \
    --hf_version "v1" \
    --hf_max_tokens 100000000 \
    --vocab_size 50257 \
    --d_model 512 \
    --n_layers 6 \
    --n_heads 8 \
    --max_seq_len 512 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 10 \
    --num_workers 4
```

### Finding Compatible Datasets

Look for datasets on HuggingFace Hub that:
1. Are pretokenized (check dataset card/description)
2. Use GPT-2 or similar tokenizer (vocab_size 50257)
3. Are stored as Parquet files
4. Have shard-based structure

**Popular options:**
- FineWeb: `HuggingFaceFW/fineweb`
- C4 (pretokenized versions)
- The Pile (pretokenized versions)

## Option 2: Simple Text File

For smaller datasets or custom data, use a simple text file:

```bash
python train.py \
    --data_file path/to/your/data.txt \
    --vocab_size 50257 \
    --max_seq_len 512
```

**File format**: One sentence per line
```
This is the first sentence.
This is the second sentence.
Another sentence here.
```

**Note**: This uses a simple hash-based tokenizer, not a real tokenizer. For best results, use pretokenized HuggingFace datasets.

## Option 3: Dummy Data (Default)

If no dataset is specified, the script uses dummy data for testing:

```bash
python train.py --vocab_size 50257
```

## Important Notes

1. **Vocabulary Size**: Make sure `--vocab_size` matches your tokenizer (e.g., 50257 for GPT-2)
2. **Sequence Length**: `--max_seq_len` should match your data's sequence length
3. **EOD Token**: The script uses `vocab_size - 1` as EOD token by default, or specify with `--eod_token`
4. **Distributed Training**: HuggingFace datasets automatically handle shard distribution across GPUs

## Troubleshooting

**Dataset not found?**
- Check the repository name is correct
- Verify the dataset is public or you're authenticated
- Check if the dataset has the required structure (Parquet shards)

**Out of memory?**
- Reduce `--batch_size`
- Use `--hf_max_tokens` to limit dataset size
- Reduce `--max_seq_len`

**Slow loading?**
- Increase `--num_workers` (but not too high)
- Check your network connection (datasets download from HuggingFace)

