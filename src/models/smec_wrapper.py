import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .ads import AdaptiveDimensionSelection

class SMECModel(nn.Module):
    """
    SMEC Model Wrapper.
    Wraps a huggingface transformer backbone and adds ADS layer.
    """
    def __init__(self, model_name: str, embedding_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load backbone
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # ADS Layer - will be initialized or updated during training
        # We start with full dimension, ADS is applied for compression
        self.ads = None # Initialized later or dynamically
        self.current_dim = embedding_dim
        
    def set_ads_target_dim(self, target_dim: int):
        """
        Initialize or update ADS layer for a specific target dimension.
        """
        self.current_dim = target_dim
        self.ads = AdaptiveDimensionSelection(self.embedding_dim, target_dim)

    def forward(self, input_ids, attention_mask):
        # Backbone forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling (standard for sentence embeddings)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize?
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        if self.ads is not None:
             compressed_embeddings = self.ads(embeddings)
             return compressed_embeddings
             
        return embeddings

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
