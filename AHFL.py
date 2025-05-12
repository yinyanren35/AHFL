# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:42:06 2025

@author: 20808
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt



#AHFL
class SelfAttention(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, d_model)
        self.key_layer = nn.Linear(input_dim, d_model)
        self.value_layer = nn.Linear(input_dim, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = torch.sum(attention_output, dim=1, keepdim=True)
        
        return attention_weights, attention_output

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

class Sero(nn.Module):
    def __init__(self):
        super(Sero, self).__init__()
        self.fc1 = nn.Linear(6670, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 15)

    def forward(self, x):
        original_x = x.clone()
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        x = x.unsqueeze(-1)
        original_x_transposed = original_x.permute(0, 2, 1)
        result = torch.bmm(original_x_transposed, x)
        final_output = result.squeeze(-1)
        return final_output, x

class Combine_Module(nn.Module):
    def __init__(self, input_dim, d_model, output_size, num_heads, hidden_size, num_layers, alpha):
        super(Combine_Module, self).__init__()
        self.attention = SelfAttention(input_dim, d_model)
        self.weight = Sero()
        self.classifier = TransformerModel(d_model, output_size, num_heads, hidden_size, num_layers)
        self.alpha = alpha  # 定义超参数 alpha
        
    def forward(self, inputs):
        attention_weights, result1 = self.attention(inputs)
        result2, x = self.weight(inputs)

        # 加权组合：result1 * alpha + result2 * (1 - alpha)
        #result1 from attention, result2 from weight, result1*aerfa, result2*(1-aerfa).
        weighted_result1 = result1 * self.alpha
        weighted_result2 = result2.unsqueeze(1) * (1 - self.alpha)
        
        # 最终组合结果
        combine_result = weighted_result1 + weighted_result2
        
        # 将组合后的结果传入分类器
        result = self.classifier(combine_result)
        
        return result, attention_weights, x.squeeze()