
from transformers import AutoModel
import torch
import torch.nn as nn
import timm
from tqdm.auto import tqdm

class MultimodalCalorieRegressor(nn.Module):
    def __init__(self,
                 text_model_name='bert-base-uncased',
                 image_model_name='resnet50',
                 emb_dim=256,
                 hidden_dim=256,
                 dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0 
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.calorie_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.image_model.parameters():
            p.requires_grad = False

        for name, param in self.image_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        for name, param in self.text_model.named_parameters():
            if ("encoder.layer.10" in name 
                or "encoder.layer.11" in name 
                or "pooler" in name):
                param.requires_grad = True


    def forward(self, batch):
        image_input = batch["image"]
        text_input = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        text_out = self.text_model(**text_input)    
        text_features = text_out.last_hidden_state[:, 0, :]
        image_features = self.image_model(image_input)
        
        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        fused_emb = torch.cat([text_emb, image_emb], dim=1)
        fused = self.fusion(fused_emb)

        mass = batch["total_mass"].unsqueeze(1)
        fused_plus_mass = torch.cat([fused, mass], dim=1)
        calories_pred = self.calorie_head(fused_plus_mass).squeeze(-1)

        return calories_pred

def evaluate(model, val_loader, epoch, device, collect_errors=False):
    model.eval()
    total_abs_err = 0.0
    n = 0
    errors_info = [] if collect_errors else None
    val_bar = tqdm(val_loader, desc=f"Валидация эпоха {epoch}")
   
    with torch.no_grad():
        for batch in val_bar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            cal_pred = model(batch)
            cal_target = batch["total_calories"]
            err = (cal_pred - cal_target).abs()
            total_abs_err += err.sum().item()
            n += cal_target.size(0)

            if collect_errors:
                dish_ids = batch["dish_id"]
                if torch.is_tensor(dish_ids):
                    dish_ids = dish_ids.cpu().tolist()
                for e, dish_id in zip(err.cpu().tolist(), dish_ids):
                    errors_info.append({
                        "dish_id": dish_id,
                        "error": float(e),
                    })

    mae = total_abs_err / n
    if collect_errors:
        errors_info_sorted = sorted(errors_info, key=lambda x: x["error"], reverse=True)
        return mae, errors_info_sorted
    return mae

def train_model(model, train_loader, val_loader, device, config):
    criterion = torch.nn.L1Loss()
    model.to(device)
    optimizer = get_optimizer(model, config)
    pred_val_mae = None
    no_improve_epochs = 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Тренировка эпоха {epoch}")

        for batch in train_bar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            cal_pred = model(batch)

            cal_target = batch["total_calories"]
            loss = criterion(cal_pred, cal_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_abs_err = (cal_pred - cal_target).abs().sum().item()
            total_train_loss += batch_abs_err
            n_train += cal_target.size(0)

        train_mae = total_train_loss / n_train
        val_mae = evaluate(model, val_loader, epoch, device)

        print(f"Эпоха {epoch} | train_MAE: {train_mae:.2f} | val_MAE: {val_mae:.2f}\n")
        if (pred_val_mae is None or val_mae < pred_val_mae):
            pred_val_mae = val_mae
            no_improve_epochs = 0
            torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_mae": val_mae,
                    },
                    config.SAVE_PATH
                )
        else:
            no_improve_epochs += 1    

        if (no_improve_epochs >= config.MAX_EPOCHS_NO_IMPROVEMENTS):
            break

def get_optimizer(model, config):
    head_params = list(model.text_proj.parameters()) + \
        list(model.image_proj.parameters()) + \
        list(model.fusion.parameters()) + \
        list(model.calorie_head.parameters())
    
    text_backbone_params = [p for p in model.text_model.parameters() if p.requires_grad]
    image_backbone_params = [p for p in model.image_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.HEAD_LR},
            {"params": text_backbone_params, "lr": config.TEXT_LR},
            {"params": image_backbone_params,"lr": config.IMAGE_LR},
        ],
        weight_decay = config.WEIGHT_DECAY,
    )
    return optimizer