#!/usr/bin/env python3
"""
Simple training script with minimal dependencies.
Uses a basic CNN encoder + LSTM decoder instead of pretrained ViT + GPT2.
"""
import argparse
import json
import os
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Simple vocabulary for CadQuery tokens
VOCAB = {
    '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
    'result': 4, '=': 5, 'cq': 6, '.': 7, 'Workplane': 8, '(': 9, ')': 10,
    '"XY"': 11, 'box': 12, 'cylinder': 13, ',': 14, '1': 15, '2': 16, '5': 17, '10': 18
}
VOCAB_SIZE = len(VOCAB)
MAX_LENGTH = 16

def tokenize_simple(code: str):
    """Simple tokenization."""
    tokens = [VOCAB['<START>']]
    for word in code.split():
        tokens.append(VOCAB.get(word, VOCAB['<UNK>']))
    tokens.append(VOCAB['<END>'])
    
    # Pad to max length
    while len(tokens) < MAX_LENGTH:
        tokens.append(VOCAB['<PAD>'])
    return tokens[:MAX_LENGTH]

class SimpleDataset(Dataset):
    def __init__(self, jsonl_path: str, image_size: int = 64):
        self.data = []
        
        # Create dummy data for testing
        for i in range(20):
            image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            code = 'result = cq Workplane "XY" box 1 1 1'
            self.data.append((image, code))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, code = self.data[idx]
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        tokens = torch.LongTensor(tokenize_simple(code))
        return image, tokens

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, MAX_LENGTH * VOCAB_SIZE)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output.view(-1, MAX_LENGTH, VOCAB_SIZE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="models/simple")
    args = parser.parse_args()
    
    print("Starting simple training with minimal dependencies...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = SimpleDataset("dummy")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB['<PAD>'])
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for images, tokens in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), tokens.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Model saved to {args.output_dir}/model.pt")

if __name__ == "__main__":
    main() 