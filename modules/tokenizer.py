import torch
import sentencepiece as spm

class TextTokenizer:
    def __init__(self, model_type, vocab_size, data_file: str, sp_model_prefix: str = 'm',
                 normalization_rule_name: str = 'nmt_nfkc'):
        
        self.vocab_size = vocab_size
        model_path = f'{sp_model_prefix}.model'

        print(f"Training SentencePiece model: {sp_model_prefix}...")
        spm.SentencePieceTrainer.train(
            input=data_file,
            model_prefix=sp_model_prefix,
            vocab_size=self.vocab_size,
            model_type=model_type,
            normalization_rule_name=normalization_rule_name,
            unk_id=0, unk_piece="[UNK]",
            bos_id=1, bos_piece="[BOS]",
            eos_id=2, eos_piece="[EOS]",
            pad_id=3, pad_piece="[PAD]"
        )

        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)
        

    def text2ids(self, texts):
        if isinstance(texts, str):
            return self.sp_model.encode(texts)
        return [self.sp_model.encode(text.strip()) for text in texts]

    def ids2text(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().tolist()
        
        if len(ids) > 0 and isinstance(ids[0], list): 
            return [self.sp_model.decode(seq) for seq in ids]
            
        return self.sp_model.decode(ids)

    @classmethod
    def from_existing(cls, model_path: str):
        instance = cls.__new__(cls)
        instance.sp_model = spm.SentencePieceProcessor(model_file=str(model_path))
        instance.vocab_size = instance.sp_model.get_piece_size()
        return instance

