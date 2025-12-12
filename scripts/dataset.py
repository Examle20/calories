import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class MultimodalDataset(Dataset):
    def __init__(self, dish, ingredients, transforms=None):
        self.dish = dish
        self.ingredients = ingredients
        self.transforms = transforms

    def __len__(self):
        return len(self.dish)

    def __getitem__(self, idx):
        row = self.dish.iloc[idx]
        total_mass = float(row["total_mass"])
        total_calories = float(row["total_calories"])
        ingredient_ids = row["ingredients"].split(";")
        ingredients = [
            self.ingredients.loc[
                self.ingredients["id"] == int(item.split("_")[1]),
                "ingr"
            ].iloc[0]
            for item in ingredient_ids
        ]
        image_path = f'data/images/{row["dish_id"]}/rgb.png'
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))["image"]
        return {
            "dish_id": row["dish_id"],
            "total_mass": total_mass,
            "total_calories": total_calories,
            "ingredients": ingredients,
            "image": image,
        }

def collate_fn(batch, tokenizer):
    total_mass = torch.tensor([item["total_mass"] for item in batch], dtype=torch.float32)
    ingredients = ["; ".join(item["ingredients"]) for item in batch]
    images = torch.stack([item["image"] for item in batch])
    total_calories = torch.tensor([item["total_calories"] for item in batch], dtype=torch.float32)
    dish_ids = [item["dish_id"] for item in batch]

    tokenized = tokenizer(
        ingredients,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    return {
        "dish_id": dish_ids,
        "total_mass": total_mass,
        "total_calories": total_calories,
        "image": images,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def draw_item(item, idx=None, mae=None):
    lines  = [
        f"Вес: {item["total_mass"]} g",
        f"Калорийность: {item["total_calories"]:.0f} kcal",
        "",
        "Ингредиенты:",
        *[f"- {ing}" for ing in item["ingredients"]],
        ""
    ]
    if idx is not None and mae is not None:
        lines.append(f"Номер: {idx}")
        lines.append(f"MAE: {mae}")
    text = "\n".join(lines)
    _, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(10, 5))

    ax_img.imshow(item["image"])
    ax_img.axis("off")

    ax_txt.axis("off")
    ax_txt.text(
        0.0, 1.0,
        text,
        va="top",
        ha="left",
        wrap=True,
    )

    plt.tight_layout()
    plt.show()

def get_data(path):
    dish = pd.read_csv("data/dish.csv")
    train_data = dish[dish["split"] == "train"].reset_index(drop=True)
    test_data = dish[dish["split"] == "test"].reset_index(drop=True)
    return train_data, test_data


IMG_SIZE = 224

train_transforms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0,
        rotate_limit=10,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5,
    ),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02,
        p=0.5,
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

def get_by_dish_id(dataset, dish_id):
    df = dataset.dish
    row_idx = df.index[df["dish_id"] == dish_id]
    idx = int(row_idx[0])
    sample = dataset[idx]
    return idx, sample