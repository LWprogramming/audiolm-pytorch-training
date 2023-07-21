import accelerate

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

accelerator = accelerate.Accelerator()

# Very small synthetic dataset
X = torch.rand(8, 4)  # 8 samples, 4 features
y = torch.randint(0, 2, (8, 1))  # binary labels

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2)

# Tiny model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

model.to(accelerator.device)  # Add this line

for i, (X_batch, y_batch) in enumerate(dataloader):
    preds = model(X_batch)
    loss = loss_fn(preds, y_batch.float())

    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    print(f"accelerator arrived on device {accelerator.device} and is main? {accelerator.is_main_process} before wait")
    accelerator.wait_for_everyone()
    print(f"accelerator arrived on device {accelerator.device} and is main? {accelerator.is_main_process} after wait")
    if accelerator.is_main_process:
        time.sleep(3)  # Sleep on main process

    if i == 1:
        break  # Just train two steps

accelerator.print('All done!')