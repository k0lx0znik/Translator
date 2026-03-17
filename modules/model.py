import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, nhead, num_layers, pad_id,
                 bos_id, eos_id, dropout=0.1):
        super().__init__()

        self.embed_size = embed_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.embedding_src = nn.Embedding(input_dim, embed_size, padding_idx=pad_id)
        self.embedding_tgt = nn.Embedding(output_dim, embed_size, padding_idx=pad_id)

        self.pos_encoder = PositionalEncoder(embed_size, dropout)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.fc_out = nn.Linear(embed_size, output_dim)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src == self.pad_id)

    def forward(self, src, tgt):
        tgt_input = tgt[:, :-1] # убираем eos
    
        src_padding_mask = self.make_src_mask(src)
        tgt_padding_mask = self.make_src_mask(tgt_input)
    
        src_padding_mask = src_padding_mask.to(torch.float32).masked_fill(src_padding_mask, float('-inf')).masked_fill(~src_padding_mask, 0.0)
        tgt_padding_mask = tgt_padding_mask.to(torch.float32).masked_fill(tgt_padding_mask, float('-inf')).masked_fill(~tgt_padding_mask, 0.0)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)

        src_emb = self.pos_encoder(self.embedding_src(src) * math.sqrt(self.embed_size))
        tgt_emb = self.pos_encoder(self.embedding_tgt(tgt_input) * math.sqrt(self.embed_size))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.fc_out(out)

    @torch.no_grad()
    def inference(self, src_tensor, tokenizer, max_len=55, beam_size=5):
        self.eval()
        device = src_tensor.device
        batch_size = src_tensor.size(0)
    
        src_padding_mask = self.make_src_mask(src_tensor)
        src_mask_inf = src_padding_mask.to(torch.float32).masked_fill(src_padding_mask, float('-inf')).masked_fill(~src_padding_mask, 0.0)
        
        src_emb = self.pos_encoder(self.embedding_src(src_tensor) * math.sqrt(self.embed_size))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask_inf)
    
        final_translations = []
    
        for b in range(batch_size):
            curr_memory = memory[b:b+1]  # (1, src_len, emb_size)
            curr_mask = src_mask_inf[b:b+1]
    
            beams = [([self.bos_id], 0.0)]
    
            for _ in range(max_len):
                all_candidates = []
                
                for seq, score in beams:
                    if seq[-1] == self.eos_id:
                        all_candidates.append((seq, score))
                        continue
    
                    tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(len(seq)).to(device)
                    tgt_emb = self.pos_encoder(self.embedding_tgt(tgt_tensor) * math.sqrt(self.embed_size))
    
                    out = self.transformer.decoder(
                        tgt_emb, 
                        curr_memory, 
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=curr_mask
                    )
                    
                    logits = self.fc_out(out[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
    
                    top_k_probs, top_k_ids = torch.topk(log_probs, beam_size)
    
                    for i in range(beam_size):
                        candidate_seq = seq + [top_k_ids[i].item()]
                        candidate_score = score + top_k_probs[i].item()
                        all_candidates.append((candidate_seq, candidate_score))
    
                beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    
                if all(seq[-1] == self.eos_id for seq, score in beams):
                    break
    
            best_seq = beams[0][0]
            
            clean_tokens = []
            for t in best_seq:
                if t == self.bos_id: 
                    continue
                if t == self.eos_id: 
                    break
                clean_tokens.append(t)
                
            final_translations.append(tokenizer.sp_model.decode(clean_tokens))
    
        return final_translations
