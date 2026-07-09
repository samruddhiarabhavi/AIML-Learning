import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv("exam_scores.csv")
df["pass"] = (df["score"] >= 70).astype(int)

X = torch.tensor(df[["hours_studied"]].values, dtype=torch.float32)
y = torch.tensor(df["pass"].values, dtype=torch.float32).reshape(-1, 1)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 4)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.output_activation(x)
        return x

model = SimpleNet()
print(model)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
