# simple_transformer_layer.py
# A minimal PyTorch implementation demonstrating a single Transformer Encoder layer.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(SimpleTransformerEncoderLayer, self).__init__()
        
        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. Self-Attention sub-layer
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)  # Add & Norm
        src = self.norm1(src)
        
        # 2. Feed Forward sub-layer
        ffn_output = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(ffn_output) # Add & Norm
        src = self.norm2(src)
        
        return src

if __name__ == '__main__':
    # Configuration
    D_MODEL = 128
    NHEAD = 4
    SEQ_LEN = 10
    BATCH_SIZE = 2

    # Initialize model
    encoder_layer = SimpleTransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD)
    
    # Create dummy input tensor (Batch, Sequence Length, Embedding Dimension)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    
    # Forward pass
    output = encoder_layer(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Simple Transformer Layer executed successfully using PyTorch.")
