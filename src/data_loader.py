from datasets import load_dataset
from pathlib import Path
import random


def load_quora_dataset(cache_dir: str = "./data_cache"):
    """Load Quora dataset with local caching.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        The train split of the Quora Question Pairs dataset
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    dataset = load_dataset("sentence-transformers/quora-duplicates", 'pair-class', cache_dir=cache_dir)
    return dataset["train"]


def sample_dataset(dataset, sample_size: int, random_seed: int | None = None):
    """Sample question pairs from dataset.
    
    Args:
        dataset: The dataset to sample from
        sample_size: Number of samples to select
        random_seed: Optional seed for reproducibility
        
    Returns:
        List of sampled examples
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    dataset_size = len(dataset)
    actual_sample_size = min(sample_size, dataset_size)
    indices = random.sample(range(dataset_size), actual_sample_size)
    samples = [dataset[i] for i in indices]
    
    return samples
