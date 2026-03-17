import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_sp_model, tgt_sp_model, max_len):

        self.src_sentences = open(src_path, encoding="utf-8").read().splitlines()
        self.tgt_sentences = open(tgt_path, encoding="utf-8").read().splitlines()

        assert len(self.src_sentences) == len(self.tgt_sentences)

        self.src_sp = src_sp_model.sp_model
        self.tgt_sp = tgt_sp_model.sp_model

        self.max_len = max_len

        self.bos_id = self.tgt_sp.bos_id()
        self.eos_id = self.tgt_sp.eos_id()
        self.pad_id = self.tgt_sp.pad_id()

    def __len__(self):
        return len(self.src_sentences)

    def encode_src(self, text):
        ids = self.src_sp.encode(text)
        return ids[:self.max_len]

    def encode_tgt(self, text):
        ids = self.tgt_sp.encode(text)
        ids = [self.bos_id] + ids + [self.eos_id]
        return ids[:self.max_len]

    def __getitem__(self, idx):

        src = self.src_sentences[idx].strip()
        tgt = self.tgt_sentences[idx].strip()

        src_ids = self.encode_src(src)
        tgt_ids = self.encode_tgt(tgt)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }