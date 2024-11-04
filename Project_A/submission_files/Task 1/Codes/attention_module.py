"""
Header for attention module.py
------------------------------

This module defines a function, `get_attention`, to compute attention maps from a given feature set.
The function calculates attention maps based on various configurations, including:
- Different parameters to control the attention calculation method (e.g., summing or maxing the absolute feature values)
- Different normalization schemes (e.g., L2, Frobenius, L1 norms) to standardize the attention maps.

The `get_attention` function can be used to apply attention in deep learning models by highlighting important 
regions within feature maps. The configurations make it flexible to fit various model architectures and objectives.

Source:
- This code is adapted from the DataDAM repository:
  Code: https://github.com/DataDistillation/DataDAM

Imports:
- `torch`: PyTorch's core library for tensor operations.
- `torch.nn.functional` (F): Functional API in PyTorch, used here for applying different types of normalization.

Function:
- `get_attention(feature_set, param=0, exp=4, norm='l2')`: 
  Computes attention maps based on the given feature set and parameters.
  - Parameters:
    - `feature_set`: The input tensor representing feature maps, expected to be a 4D tensor [B, C, H, W].
    - `param`: Controls how the attention map is computed:
      - `0`: Sum of absolute feature values.
      - `1`: Sum of absolute feature values raised to the power of `exp`.
      - `2`: Max of absolute feature values raised to the power of `exp`.
    - `exp`: Exponent applied when `param` is set to 1 or 2, adding non-linearity to the attention calculation.
    - `norm`: Type of normalization to apply to the attention map:
      - `'l2'`: L2 normalization.
      - `'fro'`: Frobenius norm.
      - `'l1'`: L1 normalization.
      - `'none'`: No normalization applied.
      - `'none-vectorized'`: No normalization, but the map is vectorized.
  - Returns:
    - `normalized_attention_maps`: The final attention map after applying the specified calculations and normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Module
def get_attention(feature_set, param=0, exp=4, norm='l2'):
    
    """
    Compute attention maps based on feature sets and specified parameters.

    Args:
        feature_set (torch.Tensor): Input tensor with feature maps [B, C, H, W].
        param (int): Determines attention calculation method (0: sum, 1: sum with exponent, 2: max with exponent).
        exp (int): Exponent applied when param is set to 1 or 2.
        norm (str): Normalization method ('l2', 'fro', 'l1', 'none', 'none-vectorized').

    Returns:
        torch.Tensor: Normalized attention maps based on input feature set and parameters.
    """
    if param==0:
        attention_map = torch.sum(torch.abs(feature_set), dim=1)
    
    elif param ==1:
        attention_map =  torch.sum(torch.abs(feature_set)**exp, dim=1)
    
    elif param == 2:
        attention_map =  torch.max(torch.abs(feature_set)**exp, dim=1)
       
    if norm == 'l2': 
        # Dimension: [B x (H*W)] -- Vectorized
        vectorized_attention_map =  attention_map.view(feature_set.size(0), -1)
        normalized_attention_maps = F.normalize(vectorized_attention_map, p=2.0)
    
    elif norm == 'fro':
        # Dimension: [B x H x W] -- Un-Vectorized
        un_vectorized_attention_map =  attention_map
        # Dimension: [B]
        fro_norm = torch.sum(torch.sum(torch.abs(attention_map)**2, dim=1), dim=1)
        # Dimension: [B x H x W] -- Un-Vectorized)
        normalized_attention_maps = un_vectorized_attention_map / fro_norm.unsqueeze(dim=-1).unsqueeze(dim=-1)
    elif norm == 'l1': 
        # Dimension: [B x (H*W)] -- Vectorized
        vectorized_attention_map =  attention_map.view(feature_set.size(0), -1)
        normalized_attention_maps = F.normalize(vectorized_attention_map, p=1.0)
    
    elif norm =='none':
        normalized_attention_maps = attention_map
        
    elif norm == 'none-vectorized':
        normalized_attention_maps =  attention_map.view(feature_set.size(0), -1)
    
    return normalized_attention_maps

