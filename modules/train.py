import torch
import sacrebleu
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn as nn

def training_epoch(model, optimizer, criterion, loader, device, tqdm_desc, scaler, scheduler):
    model.train()
    train_loss = 0.0

    for batch in tqdm(loader, desc=tqdm_desc):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        
        optimizer.zero_grad()

        with autocast():
            output = model(src, tgt) 
            
            output = output.reshape(-1, output.shape[-1])
            targets = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        train_loss += loss.item() * src.size(0)

    return train_loss / len(loader.dataset)

@torch.no_grad()
def validation_epoch(model, criterion, loader, device, tqdm_desc):
    model.eval()
    val_loss = 0.0

    for batch in tqdm(loader, desc=tqdm_desc):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        with autocast():
            output = model(src, tgt)
            
            output = output.reshape(-1, output.shape[-1])
            targets = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, targets)
        
        val_loss += loss.item() * src.size(0)

    return val_loss / len(loader.dataset)

def quick_bleu(model, loader, tokenizer, device, num_batches=2):
    model.eval()

    preds = []
    refs = []

    model_for_infer = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

            src_tensor = batch["src"].to(device)
            tgt_tensor = batch["tgt"]

            translations = model_for_infer.inference(src_tensor, tokenizer)
            preds.extend(translations)
            
            for ref in tgt_tensor:
                tokens = ref.tolist()
                
                special_ids = {
                    tokenizer.sp_model.pad_id(),
                    tokenizer.sp_model.bos_id(),
                    tokenizer.sp_model.eos_id()
                }
                
                clean = [t for t in tokens if t not in special_ids]
                
                reference_text = tokenizer.sp_model.decode(clean)
                refs.append(reference_text)

    bleu = sacrebleu.corpus_bleu(preds, [refs])

    return bleu.score

def train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, tokenizer_tgt, device):

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.sp_model.pad_id(),
        label_smoothing=0.1
    )

    scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):

        train_loss = training_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            f'Training {epoch}/{num_epochs}',
            scaler,
            scheduler
        )

        val_loss = validation_epoch(
            model,
            criterion,
            val_loader,
            device,
            f'Validating {epoch}/{num_epochs}'
        )

        bleu = quick_bleu(model, val_loader, tokenizer_tgt, device)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.2f}')