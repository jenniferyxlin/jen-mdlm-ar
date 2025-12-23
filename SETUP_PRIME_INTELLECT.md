# Setting Up MDLM on Prime Intellect GPU Nodes

This guide will help you clone and run this repository on a Prime Intellect node with GPU support.

## Your Node Configuration

Based on your Prime Intellect node:
- **GPUs:** 8 x NVIDIA H100 80GB
- **OS:** Ubuntu 22
- **CUDA:** 12.x
- **vCPUs:** 104
- **Memory:** 752 GB
- **Disk:** 10.84 TB

## Prerequisites

- Access to your Prime Intellect node (Ubuntu 22, CUDA 12)
- Git installed on the node (usually pre-installed)
- Python 3.8+ installed (Ubuntu 22 typically comes with Python 3.10+)

## Step 1: Clone the Repository

Once you're connected to your Prime Intellect node, clone the repository:

```bash
# Clone the repository
git clone <your-repo-url> jen_mdlm
cd jen_mdlm
```

If you're using GitHub, it would look like:
```bash
git clone https://github.com/yourusername/jen_mdlm.git
cd jen_mdlm
```

## Step 2: Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

## Step 3: Install Dependencies

Your node has CUDA 12, so install PyTorch with CUDA 12.1 support (compatible with CUDA 12):

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (compatible with your CUDA 12)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

**Note:** Prime Intellect nodes may come with PyTorch pre-installed. First check:
```bash
# Check CUDA version and GPU status
nvidia-smi

# Check if PyTorch is already installed and can see GPUs
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

If PyTorch is already installed with CUDA support and can see all 8 GPUs, you can skip the PyTorch installation step and just install the other dependencies:
```bash
pip install -r requirements.txt
```

## Step 4: Verify GPU Access

Verify that PyTorch can see all 8 H100 GPUs:

```bash
# Quick check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Detailed GPU info
nvidia-smi
```

You should see 8 GPUs listed. With H100 80GB GPUs, you can use larger batch sizes and models.

## Step 5: Run Training

### Single GPU Training

For a single GPU setup:

```bash
python train.py \
    --epochs 10 \
    --batch_size 32 \
    --use_rope \
    --device cuda
```

### Multi-GPU Distributed Training (Recommended for 8x H100)

With 8 H100 GPUs, you should use distributed training for maximum performance:

```bash
# Using torchrun (recommended) - Use all 8 GPUs
torchrun --nproc_per_node=8 train.py \
    --distributed \
    --epochs 10 \
    --batch_size 32 \
    --use_rope

# Or use a subset (e.g., 4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --distributed \
    --epochs 10 \
    --batch_size 32 \
    --use_rope

# Or using python -m torch.distributed.launch (older method)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
    --distributed \
    --epochs 10 \
    --batch_size 32 \
    --use_rope
```

**Note:** With 8x H100 80GB GPUs, you can use much larger batch sizes. Try increasing `--batch_size` to 64, 128, or even higher depending on your model size.

### Using HuggingFace Datasets

You can train on HuggingFace datasets:

```bash
# Pretokenized dataset
python train.py \
    --hf_repo "flappingairplanes/fineweb-edu-sample-10bt_gpt2" \
    --hf_split "train" \
    --epochs 10 \
    --batch_size 32 \
    --use_rope \
    --device cuda

# Raw dataset with tokenization
python train.py \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-103-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs 10 \
    --batch_size 32 \
    --use_rope \
    --device cuda
```

### Custom Configuration

You can customize model parameters:

```bash
python train.py \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --max_seq_len 512 \
    --batch_size 32 \
    --epochs 10 \
    --use_rope \
    --rope_theta 10000.0 \
    --rope_percent 1.0 \
    --device cuda
```

## Step 6: Monitor Training

Monitor GPU usage during training:

```bash
# In a separate terminal, watch GPU usage
watch -n 1 nvidia-smi
```

## Common Issues and Solutions

### Issue: CUDA out of memory
**Solution:** Reduce batch size or sequence length:
```bash
python train.py --batch_size 16 --max_seq_len 256 --device cuda
```

### Issue: Multiple processes error
**Solution:** Make sure you're using the correct distributed launch command, or use single GPU mode:
```bash
python train.py --device cuda  # Single GPU
```

### Issue: Dependencies not found
**Solution:** Make sure your virtual environment is activated and all dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Prime Intellect Specific Notes

1. **Storage:** With 10.84 TB of disk space, you have plenty of room for datasets. However, for very large datasets, you can still use streaming:
   ```bash
   python train.py --hf_raw_repo "dataset" --hf_streaming --device cuda
   ```

2. **SSH Access:** Connect to your node via SSH:
   ```bash
   # Quick connect (uses SSH config)
   ssh prime-intellect-box0
   
   # Or manually:
   ssh -i ~/Documents/Primekeys/private_key.pem -p 25419 root@159.26.81.14
   ```
   Your SSH config is already set up, so just run `ssh prime-intellect-box0` to reconnect.

3. **Checkpointing:** The training script saves checkpoints. With 10.84 TB disk, you have ample space for model checkpoints. Make sure you have write permissions in the working directory.

4. **Cost Optimization:** At $14.40/hr, consider:
   - Using all 8 GPUs efficiently with distributed training
   - Monitoring training progress to avoid unnecessary runtime
   - Saving checkpoints regularly in case you need to stop and resume

5. **H100 Performance:** H100 GPUs are extremely fast. You can:
   - Use larger batch sizes (128-256+ per GPU)
   - Train larger models
   - Use longer sequences (1024-2048 tokens)
   - Enable mixed precision training if supported (check train.py for `--fp16` or `--bf16` flags)

## Quick Start Example

Here's a complete example optimized for your 8x H100 setup:

```bash
# 1. Clone and enter directory
git clone <your-repo-url> jen_mdlm
cd jen_mdlm

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
# Check if PyTorch is pre-installed first, then:
pip install -r requirements.txt
# If PyTorch not installed: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verify all 8 GPUs are visible
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 5. Run training with all 8 GPUs (recommended)
torchrun --nproc_per_node=8 train.py \
    --distributed \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-2-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs 5 \
    --batch_size 64 \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --max_seq_len 512 \
    --use_rope

# Or single GPU test first
python train.py \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-2-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs 5 \
    --batch_size 32 \
    --d_model 512 \
    --n_layers 6 \
    --n_heads 8 \
    --max_seq_len 256 \
    --use_rope \
    --device cuda
```

**Performance Tips for H100 GPUs:**
- With 80GB per GPU, you can use much larger models and batch sizes
- Try `--batch_size 128` or higher for distributed training
- Increase `--max_seq_len` to 1024 or 2048 if needed
- Use `--num_workers` to parallelize data loading (e.g., `--num_workers 8`)

## Additional Resources

- See `README.md` for more training options
- See `DATASET_USAGE.md` for detailed dataset configuration
- Check `default.yaml` for default configuration values

