# Translator
Developed a neural machine translation system based on the Transformer architecture for sequence-to-sequence language translation tasks.

A compact neural machine translation project based on the **Transformer** architecture (PyTorch) for sequence-to-sequence tasks.

## Repository structure

- `modules/model.py` — Transformer model + beam-search inference.
- `modules/tokenizer.py` — SentencePiece tokenizer utilities.
- `modules/dataset.py` — `TranslationDataset` for parallel text pairs.
- `modules/train.py` — training/validation loops and quick BLEU evaluation.
- `main.py` — CLI entrypoint for inference (single text translation).
- `modules/requirement.txt` — project dependencies.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r modules/requirement.txt
```

## Data format

`TranslationDataset` expects **two text files** with the same number of lines:

- `src.txt` — source sentences (one sentence per line)
- `tgt.txt` — target translations (one sentence per line)

Lines must be aligned by index (line `i` in `src.txt` must match line `i` in `tgt.txt`).

## Tokenization

The project uses SentencePiece via `TextTokenizer`.

- If a `*.model` file already exists, it can be loaded directly without retraining.
- If the model file does not exist, you can train it automatically (through `TextTokenizer(...)`) on a text file.

## Training (basic example)

There is no dedicated `train_main.py` script yet, but all building blocks are available in `modules/`.
A minimal pipeline example:

```python
import torch
from torch.utils.data import DataLoader

from modules.dataset import TranslationDataset
from modules.model import TransformerModel
from modules.tokenizer import TextTokenizer
from modules.train import train

# 1) Tokenizers
src_tok = TextTokenizer(model_type="bpe", vocab_size=16000, data_file="src.txt", sp_model_prefix="src_sp")
tgt_tok = TextTokenizer(model_type="bpe", vocab_size=16000, data_file="tgt.txt", sp_model_prefix="tgt_sp")

# 2) Dataset/loaders
dataset = TranslationDataset("src.txt", "tgt.txt", src_tok, tgt_tok, max_len=55)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# For demonstration purposes, reuse the same loader for validation
train_loader = loader
val_loader = loader

# 3) Model
model = TransformerModel(
    input_dim=src_tok.sp_model.get_piece_size(),
    output_dim=tgt_tok.sp_model.get_piece_size(),
    embed_size=512,
    nhead=8,
    num_layers=6,
    pad_id=tgt_tok.sp_model.pad_id(),
    bos_id=tgt_tok.sp_model.bos_id(),
    eos_id=tgt_tok.sp_model.eos_id(),
)

# 4) Optimizer/scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

# 5) Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
train(model, optimizer, scheduler, train_loader, val_loader, num_epochs=10, tokenizer_tgt=tgt_tok, device=device)
```

## Inference

`main.py` supports the `--translate` flag (alias for `--text`).

Minimal run example:

```bash
python main.py --translate "Your text here"
```

By default, the script expects:

- `checkpoint.pt`
- `src_sp.model`
- `tgt_sp.model`

If your files have different names/paths, pass explicit arguments:

```bash
python main.py \
  --translate "Your text here" \
  --checkpoint path/to/model.pt \
  --src-model-prefix path/to/src_sp \
  --tgt-model-prefix path/to/tgt_sp
```

If tokenizer `*.model` files are missing, you can provide text for automatic training:

```bash
python main.py \
  --translate "Your text here" \
  --spm-train-data data/all_texts.txt
```

## Notes

- For correct inference, model hyperparameters in `main.py` (`embed-size`, `nhead`, `num-layers`) must match those used during checkpoint training.
- If `torch` is not installed in your environment, the script will not start.
