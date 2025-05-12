# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:35:40 2024

@author: 20808
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# Testing function with additional metrics
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=False)  
            probs = torch.softmax(output, dim=1)[:, 1]  
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)

    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc_score:.4f}")

    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ABIDE')
    plt.legend(loc="lower right")
    plt.show()

# Model architecture with batch normalization and dropout
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        output = output.mean(dim=0)
        output = self.fc(output)
        return output

# Define a custom dataset for brain region data
class BrainRegionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

if __name__ == '__main__':
    
    seed = 0
    set_seed(seed)

    data_path = './Data/SITE20.mat'
    mat_data = scipy.io.loadmat(data_path)
    
    data = []
    raw_data = np.array(mat_data['AAL'])[0]
    for item in raw_data:
        data.append(item)
    data = np.array(np.stack(data, axis=0), np.float32)
    labels = np.array(mat_data['lab'][0])

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    data = data.reshape(data.shape[0], -1)  # Flatten each sample to (175*116) shape
    data = scaler.fit_transform(data)
    data = data.reshape(data.shape[0], 232, 116)  # Reshape back to (184, 175, 116)

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed)

    # Create DataLoader for training and testing sets
    train_dataset = BrainRegionDataset(train_data, train_labels)
    test_dataset = BrainRegionDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, generator=torch.Generator().manual_seed(seed))

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    input_size = 116
    output_size = 2  # Binary classification (disease, no disease)
    num_heads = 4
    hidden_size = 128
    num_layers = 2

    model = TransformerModel(input_size, output_size, num_heads, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, device, train_loader, criterion, optimizer, epochs=200)

    # Test the model
    test(model, device, test_loader, criterion)