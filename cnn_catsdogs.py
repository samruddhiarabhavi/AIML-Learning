import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=r"C:\Users\HP\aimllearn\training_data", transform=transform)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

image, label = dataset[0]
print("Image shape:", image.shape)
print("Label:", label)
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Train size:", len(train_data))
print("Test size:", len(test_data))