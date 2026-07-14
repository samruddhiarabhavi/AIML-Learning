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