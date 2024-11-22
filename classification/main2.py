import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dvclive import Live
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 16
learning_rate = 0.0001
optimizer_weight = 1e-5  

TRAIN_DATA_PATH = "/home/an/spbu_deep_learning/classification/splitted"
CSV_PATH = "/home/an/spbu_deep_learning/classification/new_annotations.csv"
LABELS_MAPPING = {
    "0": "Ace", "1": "Akainu", "2": "Brook", "3": "Chopper", "4": "Crocodile",
    "5": "Franky", "6": "Jinbei", "7": "Kurohige", "8": "Law", "9": "Luffy",
    "10": "Mihawk", "11": "Nami", "12": "Rayleigh", "13": "Robin", "14": "Sanji",
    "15": "Shanks", "16": "Usopp", "17": "Zoro"
}

class OnePieceDataset(Dataset):
    def __init__(self, images_dir, csv_path=None, labels_json=None, transform=None, split=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.is_train = csv_path is not None

        if self.is_train:
            import pandas as pd
            self.data = pd.read_csv(csv_path)

            if split:
                self.data = self.data[self.data['split'] == split]

            self.label_map = labels_json if isinstance(labels_json, dict) else None

            for _, row in self.data.iterrows():
                relative_path = row['image_path'].replace("\\", "/")
                image_path = os.path.join(images_dir, relative_path)
                image_path = os.path.normpath(image_path)
                self.image_paths.append(image_path)
                self.labels.append(row['label'])

        else:
            self.image_paths = [
                os.path.join(images_dir, fname)
                for fname in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, fname))
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            return image, image_name

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = OnePieceDataset(
    images_dir=TRAIN_DATA_PATH,
    csv_path=CSV_PATH,
    labels_json=LABELS_MAPPING,
    transform=transform,
    split="train"
)

val_dataset = OnePieceDataset(
    images_dir=TRAIN_DATA_PATH,
    csv_path=CSV_PATH,
    labels_json=LABELS_MAPPING,
    transform=transform,
    split="val"
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(LABELS_MAPPING))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=optimizer_weight)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    current_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()
        _, label = preds.max(1)
        correct += label.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = current_loss / len(dataloader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    current_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            loss = loss_fn(preds, labels)

            current_loss += loss.item()
            _, label = preds.max(1)
            correct += label.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = current_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

with Live() as live:  
    live.log_param("epochs", epochs)
    live.log_param("batch_size", batch_size)
    live.log_param("learning_rate", learning_rate)
    live.log_param("weight_decay", optimizer_weight)

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)
        live.log_metric("train_loss", train_loss)
        live.log_metric("train_acc", train_acc)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        live.log_metric("val_loss", val_loss)
        live.log_metric("val_acc", val_acc)

        live.next_step()  
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%")
        print(f"val loss: {val_loss:.4f}, val accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), "one_piece_model.pth")