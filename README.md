# Transformer Attention from Scratch

Name: Ainavole Hasini Reddy  
Student ID: 700773828

## Overview

This repository contains implementations of fundamental transformer components built from scratch using NumPy and PyTorch.

## Contents

### 1. Scaled Dot-Product Attention (NumPy)
Implementation of the attention mechanism using pure NumPy. Includes softmax normalization and returns both attention weights and context vectors. Supports optional masking for causal attention.

### 2. Transformer Encoder Block (PyTorch)
Complete transformer encoder block with multi-head self-attention, position-wise feed-forward network, residual connections, and layer normalization.

## Requirements

```bash
numpy
torch
scikit-learn
```

## Sample Outputs

### Scaled Dot-Product Attention Output

```
Input Shapes:
Q shape: (2, 4, 8)
K shape: (2, 4, 8)
V shape: (2, 4, 8)

Output Shapes:
Context shape: (2, 4, 8)
Attention weights shape: (2, 4, 4)

Attention Weights (first batch):
[[0.50258161 0.07125796 0.29559999 0.13056044]
 [0.1356621  0.4668456  0.09868613 0.29880617]
 [0.04296256 0.73176166 0.07266276 0.15261301]
 [0.54250442 0.19083811 0.14071448 0.12594299]]

Sum of attention weights (should be ~1.0 for each query):
[1. 1. 1. 1.]

Context Vector (first batch, first query):
[-0.01769307 -0.01455707 -1.04874428 -0.53684339 -0.11980371  0.48081568
 -0.65812311  0.98025517]

Example with Causal Masking:
Causal Attention Weights (first batch):
[[1.         0.         0.         0.        ]
 [0.22516244 0.77483756 0.         0.        ]
 [0.05070005 0.86355075 0.0857492  0.        ]
 [0.54250442 0.19083811 0.14071448 0.12594299]]
```

### Transformer Encoder Block Output

```
Transformer Encoder Block - Configuration
Batch size: 32
Sequence length: 10
Model dimension (d_model): 512
Number of heads: 8
Feed-forward dimension (d_ff): 2048
Dropout rate: 0.1

Input shape: torch.Size([32, 10, 512])

Output Verification
Output shape: torch.Size([32, 10, 512])
Expected shape: torch.Size([32, 10, 512])

Model Architecture Summary
TransformerEncoderBlock(
  (attention): MultiHeadAttention(
    (W_q): Linear(in_features=512, out_features=512, bias=True)
    (W_k): Linear(in_features=512, out_features=512, bias=True)
    (W_v): Linear(in_features=512, out_features=512, bias=True)
    (W_o): Linear(in_features=512, out_features=512, bias=True)
  )
  (ffn): FeedForwardNetwork(
    (linear1): Linear(in_features=512, out_features=2048, bias=True)
    (linear2): Linear(in_features=2048, out_features=512, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (dropout1): Dropout(p=0.1, inplace=False)
  (dropout2): Dropout(p=0.1, inplace=False)
)

Parameter Count
Total parameters: 3,152,384
Trainable parameters: 3,152,384
```

