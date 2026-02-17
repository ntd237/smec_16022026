import logging
from typing import List, Dict, Union, Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)

class SMECDataset(Dataset):
    """
    Dataset wrapper for SMEC training.
    Supports loading data from MTEB/BEIR or custom HuggingFace datasets.
    """
    def __init__(
        self, 
        dataset_name: str, 
        split: str = "train", 
        max_length: int = 512,
        tokenizer = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Load dataset
        # For simplicity, we assume standard MTEB/BEIR format or 'sentence-transformers' format
        # This is a placeholder for actual data loading logic which can be complex depending on the dataset
        try:
            self.data = load_dataset(dataset_name, split=split)
            logger.info(f"Loaded {len(self.data)} samples from {dataset_name}/{split}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Placeholder for retrieving a sample
        # Returns: {'query': str, 'positive': str, 'negative': List[str]}
        item = self.data[idx]
        
        # Basic parsing logic - needs adjustment based on specific dataset structure (e.g., Quora, MSMARCO)
        query = item.get('query', '') or item.get('sentence1', '')
        positive = item.get('positive', '') or item.get('sentence2', '') # Assuming pairs for now
        
        # Handle negatives if available, else empty list
        negatives = item.get('negatives', [])
        
        return {
            'query': query,
            'positive': positive,
            'negatives': negatives
        }

def collate_fn(batch):
    """
    Collate function to prepare batch for the model.
    """
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negatives'] for item in batch]
    
    return {
        'queries': queries,
        'positives': positives,
        'negatives': negatives
    }

def get_dataloader(
    dataset_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Factory function to create DataLoader for SMEC training.
    """
    dataset = SMECDataset(dataset_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
