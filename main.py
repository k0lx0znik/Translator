import argparse
from pathlib import Path

import torch

from modules.model import TransformerModel
from modules.tokenizer import TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for Transformer translator")
    parser.add_argument("--text", type=str, required=True, help="Source text to translate")
    parser.add_argument("--src-model-prefix", type=str, required=True, help="Prefix of source SentencePiece model")
    parser.add_argument("--tgt-model-prefix", type=str, required=True, help="Prefix of target SentencePiece model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--spm-train-data", type=str, required=True, help="Text file path for tokenizer initialization")
    parser.add_argument("--model-type", type=str, default="bpe", help="SentencePiece model type")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Tokenizer vocabulary size")
    parser.add_argument("--embed-size", type=int, default=512, help="Transformer embedding size")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--max-len", type=int, default=55, help="Maximum generated sequence length")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_tokenizer(prefix: str, model_type: str, vocab_size: int, train_data: str) -> TextTokenizer:
    model_path = Path(f"{prefix}.model")
    if model_path.exists():
        return TextTokenizer.from_existing(model_path)
    return TextTokenizer(
        model_type=model_type,
        vocab_size=vocab_size,
        data_file=train_data,
        sp_model_prefix=prefix,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tokenizer_src = build_tokenizer(
        prefix=args.src_model_prefix,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        train_data=args.spm_train_data,
    )
    tokenizer_tgt = build_tokenizer(
        prefix=args.tgt_model_prefix,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        train_data=args.spm_train_data,
    )

    model = TransformerModel(
        input_dim=tokenizer_src.sp_model.get_piece_size(),
        output_dim=tokenizer_tgt.sp_model.get_piece_size(),
        embed_size=args.embed_size,
        nhead=args.nhead,
        num_layers=args.num_layers,
        pad_id=tokenizer_tgt.sp_model.pad_id(),
        bos_id=tokenizer_tgt.sp_model.bos_id(),
        eos_id=tokenizer_tgt.sp_model.eos_id(),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    src_ids = tokenizer_src.text2ids(args.text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    translation = model.inference(
        src_tensor=src_tensor,
        tokenizer=tokenizer_tgt,
        max_len=args.max_len,
        beam_size=args.beam_size,
    )[0]

    print(translation)


if __name__ == "__main__":
    main()
