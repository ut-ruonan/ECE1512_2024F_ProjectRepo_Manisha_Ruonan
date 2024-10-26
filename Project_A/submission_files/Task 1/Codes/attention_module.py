import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Module
def get_attention(feature_set, param=0, exp=4, norm='l2'):
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

