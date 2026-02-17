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
            if dataset_name == 'quora':
                 # Use the official HF 'quora' dataset with trust_remote_code=True because it requires execution of a python script
                 # However, since scripts are deprecated/blocked often, we switch to a safer processed version if possible.
                 # Let's try 'sentence-transformers/quora-duplicates', but that might need authentication or different structure.
                 # Let's stick with 'quora' but enable trust_remote_code=True as a first attempt.
                 # Actually, the error `Dataset scripts are no longer supported` suggests strict blocking.
                  # We'll switch to 'sentence-transformers/quora-duplicates' which is a standard alternative.
                  # MUST specify config name 'pair' as per error message
                  self.data = load_dataset("sentence-transformers/quora-duplicates", "pair", split=split)
                  # This dataset has 'questions' (list of text) and 'is_duplicate' (bool)
                  # self.data = self.data.filter(lambda x: x['is_duplicate'])
                  pass
            else:
                self.data = load_dataset(dataset_name, split=split)
            
            if len(self.data) == 0:
                 raise ValueError("Dataset is empty after loading/filtering.")
                 
            logger.info(f"Loaded {len(self.data)} samples from {dataset_name}/{split}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            # Fallback for debugging - create dummy data? No, better to fail hard.
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Placeholder for retrieving a sample
        # Returns: {'query': str, 'positive': str, 'negative': List[str]}
        item = self.data[idx]
        
        if self.dataset_name == 'quora':
             # structure from 'sentence-transformers/quora-duplicates' ('pair' config):
             # {'anchor': str, 'positive': str}
             query = item['anchor']
             positive = item['positive']
             negatives = []
        else:
             # Basic parsing logic - needs adjustment based on specific dataset structure (e.g., Quora, MSMARCO)
             query = item.get('query', '') or item.get('sentence1', '')
             positive = item.get('positive', '') or item.get('sentence2', '') # Assuming pairs for now
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
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: int = 128
) -> DataLoader:
    """
    Factory function to create DataLoader for SMEC training.
    """
    dataset = SMECDataset(dataset_name, split=split, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
