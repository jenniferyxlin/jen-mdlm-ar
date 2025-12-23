# MDLM Implementation 
- Bidirectional attention
- Weighted masked cross-entropy loss 

## Features 
### Rotary Position Embeddings (RoPE)
- Encodes position by rotating the query/key vectors by an angle based on their position 
- For query and key in attention layers 
- Configurable `rope_theta` and `rope_percent` parameters

### EOD Token Handling 
- EOD is a structural marker, not a prediction target (including it can bias the model toward predicting EOD)
- Mask end-of-document tokens from loss computation 
- Removes lask token before masking 

### Distributed Training 
- PyTorch DistributedDataParallel (DDP) support 
- Multi-GPU training with data parallelism 
- Automtic process group initialization 

### Other 
- Substitute RMSNorm for LayerNorm, no bias terms in linear layers 
- SwiGLU activations in feed-forward networks instead of GELU
- automatic computation of FFN size as `h_f = floor(8 * d_model / (3 * 64)) * 64`

## Usage 
### Single GPU Training
```bash
python train.py --epochs 10 --batch_size 32 --use_rope
```

### Multi-GPU Distributed Training
```bash
python train.py --distributed --epochs 10 --batch_size 32 --use_rope
```

### Custom Data
```bash
python train.py --data_file your_data.txt --epochs 20 --use_rope
```

### Raw HuggingFace Datasets
Use raw text datasets from HuggingFace Hub with automatic tokenization:
```bash
python train.py \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-103-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs 10 \
    --use_rope
```
- Load the raw dataset from HuggingFace
- Tokenize it on-the-fly using the specified tokenizer
- Automatically set vocab_size from the tokenizer

### RoPE Configuration
```bash
python train.py \
    --use_rope \
    --rope_theta 10000.0 \
    --rope_percent 1.0 \
    --d_model 768 \
    --n_heads 12
```

### Automatic FFN Size (Paper Formula)
```bash
# d_ff=None uses paper formula: h_f = floor(8 * d_model / (3 * 64)) * 64
python train.py --d_model 512 --d_ff None
```