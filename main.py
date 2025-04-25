import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel

data_dir = '/home/phd/datasets/breakHis/'
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item()}')


if __name__ == "__main__":
    #data_dir ='/home/phd/datasets/breakHis/'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = BreakHisDataset(data_dir, transform)
    train_loader = dataset.get_loader()

    model = MagNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    train_model(model, train_loader, criterion, optimizer, epochs)
    