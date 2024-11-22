import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = label_map

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
                    if self.label_map:
                        label_name = os.path.basename(root)
                        self.labels.append(self.label_map.get(label_name, -1))
                    else:
                        self.labels.append(-1) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, os.path.splitext(os.path.basename(img_path))[0]

TRAIN_DATA_PATH = "/home/an/spbu_deep_learning/classification/splitted/train"
TEST_DATA_PATH = "/home/an/spbu_deep_learning/classification/splitted/test"

LABELS_MAPPING = {
    "Ace": 0, "Akainu": 1, "Brook": 2, "Chopper": 3, "Crocodile": 4,
    "Franky": 5, "Jinbei": 6, "Kurohige": 7, "Law": 8, "Luffy": 9,
    "Mihawk": 10, "Nami": 11, "Rayleigh": 12, "Robin": 13, "Sanji": 14,
    "Shanks": 15, "Usopp": 16, "Zoro": 17
}

train_dataset = CustomImageDataset(root_dir=TRAIN_DATA_PATH, transform=transform, label_map=LABELS_MAPPING)
test_dataset = CustomImageDataset(root_dir=TEST_DATA_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.regnet_x_400mf(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(LABELS_MAPPING))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/3: Loss: {loss.item():.4f}")

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for inputs, _, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        predictions.extend(preds.cpu().numpy())
        image_names.extend(filenames)

predictions_df = pd.DataFrame({
    'id': image_names,  
    'label': predictions
})
predictions_df.to_csv('submission.csv', index=False)
print("Results saved to submission.csv")
