# from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import timm
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill
# from transformers import AutoTokenizer


class MultimodalDataset(Dataset):
    def __init__(self, dish, ingredients, text_model, image_model, transforms):
        self.dish = dish
        self.ingredients = ingredients
        # self.image_cfg = timm.get_pretrained_cfg(image_model)
        # self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        total_mass = self.dish.loc[idx, "total_mass"]
        total_calories = self.dish.loc[idx, "total_calories"]
        ingredients = [self.ingredients.loc[self.ingredients["id"] == int(item.split("_")[1]), "ingr"].iloc[0] for item in self.dish.loc[idx, "ingredients"].split(';')]
        image = Image.open(f"data/images/{self.dish.loc[idx, "dish_id"]}/rgb.png").convert('RGB')
        if (self.transforms != None):
            image = self.transforms(image=np.array(image))["image"]
        # label = self.df.loc[idx, "label"]
        # img_path = self.df.loc[idx, "image_path"]
        # image = Image.open(f"data/images/{img_path}").convert('RGB')
        # image = self.transforms(image=np.array(image))["image"]

        return {"total_mass": total_mass, "total_calories": total_calories, "ingredients": ingredients, "image": image }

def collate_fn(batch):
    print(batch)
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.LongTensor([item["label"] for item in batch])

    tokenized_input = {"input_ids": []}
    return {
        "label": labels,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }

def draw_item(item):
    lines  = [
        f"Вес: {item["total_mass"]} g",
        f"Калорийность: {item["total_calories"]} kcal",
        "",
        "Ингредиенты:",
        *[f"- {ing}" for ing in item["ingredients"]],
    ]
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