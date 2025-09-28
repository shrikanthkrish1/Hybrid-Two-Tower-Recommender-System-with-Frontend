# model/dataset_fixed.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import warnings

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 usr_features: List[str], 
                 movie_features: List[str], 
                 rating_col: str,
                 tokenizer, 
                 max_len: int = 32, 
                 user2idx: Optional[Dict] = None, 
                 movie2idx: Optional[Dict] = None,
                 is_training: bool = True):
        """
        Fixed CustomDataset that handles mapping conflicts properly
        
        Args:
            df: DataFrame with ratings data
            usr_features: [user_id_col, user_tag_col]  
            movie_features: [movie_id_col, movie_genre_col]
            rating_col: name of rating column
            tokenizer: HuggingFace tokenizer
            max_len: max sequence length for tokenization
            user2idx: user to index mapping (MUST be provided for test/val sets)
            movie2idx: movie to index mapping (MUST be provided for test/val sets)  
            is_training: whether this is training dataset
        """
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_training = is_training
        
        # Get column data
        self.user_ids_raw = df[usr_features[0]].values
        self.movie_ids_raw = df[movie_features[0]].values
        self.ratings = df[rating_col].values
        
        # Handle optional tag/genre columns
        self.tags = (df[usr_features[1]].fillna("").values 
                    if len(usr_features) > 1 and usr_features[1] in df.columns 
                    else [""]*len(df))
                    
        self.genres = (df[movie_features[1]].fillna("").values 
                      if len(movie_features) > 1 and movie_features[1] in df.columns 
                      else [""]*len(df))
        
        # Handle mappings properly
        if user2idx is not None and movie2idx is not None:
            # Use provided mappings (for test/validation)
            self.user2idx = user2idx
            self.movie2idx = movie2idx
            self._validate_mappings(df, usr_features[0], movie_features[0])
        else:
            # Create new mappings (only for training)
            if not is_training:
                raise ValueError("user2idx and movie2idx must be provided for non-training datasets!")
            
            self.user2idx = {u: i for i, u in enumerate(sorted(df[usr_features[0]].unique()))}
            self.movie2idx = {m: i for i, m in enumerate(sorted(df[movie_features[0]].unique()))}
        
        # Filter out unknown users/movies if not training
        if not is_training:
            self._filter_unknown_entities(df, usr_features[0], movie_features[0])
            
        print(f"Dataset created: {len(self)} samples, {len(self.user2idx)} users, {len(self.movie2idx)} movies")

    def _validate_mappings(self, df: pd.DataFrame, user_col: str, movie_col: str):
        """Validate that mappings cover the data adequately"""
        unknown_users = set(df[user_col].unique()) - set(self.user2idx.keys())
        unknown_movies = set(df[movie_col].unique()) - set(self.movie2idx.keys())
        
        if unknown_users:
            print(f"WARNING: {len(unknown_users)} unknown users will be mapped to UNK token")
            print(f"Unknown users sample: {list(unknown_users)[:5]}")
            
        if unknown_movies:
            print(f"WARNING: {len(unknown_movies)} unknown movies will be mapped to UNK token")  
            print(f"Unknown movies sample: {list(unknown_movies)[:5]}")

    def _filter_unknown_entities(self, df: pd.DataFrame, user_col: str, movie_col: str):
        """Filter out samples with unknown users or movies (for test sets)"""
        initial_len = len(self.user_ids_raw)
        
        # Create boolean mask for known entities
        known_users_mask = np.array([u in self.user2idx for u in self.user_ids_raw])
        known_movies_mask = np.array([m in self.movie2idx for m in self.movie_ids_raw])
        valid_mask = known_users_mask & known_movies_mask
        
        # Apply filter
        self.user_ids_raw = self.user_ids_raw[valid_mask]
        self.movie_ids_raw = self.movie_ids_raw[valid_mask]
        self.ratings = self.ratings[valid_mask]
        self.tags = np.array(self.tags)[valid_mask] if isinstance(self.tags, (list, np.ndarray)) else self.tags
        self.genres = np.array(self.genres)[valid_mask] if isinstance(self.genres, (list, np.ndarray)) else self.genres
        
        filtered_count = initial_len - len(self.user_ids_raw)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} samples with unknown users/movies")

    def __len__(self):
        return len(self.user_ids_raw)

    def __getitem__(self, idx):
        user_raw = self.user_ids_raw[idx]
        movie_raw = self.movie_ids_raw[idx]

        # Map to indices with proper unknown handling
        user_id = self.user2idx.get(user_raw, 0)  # 0 can be reserved for UNK
        movie_id = self.movie2idx.get(movie_raw, 0)  # 0 can be reserved for UNK

        # Convert to tensors (proper import)
        user_id = torch.tensor(user_id, dtype=torch.long)
        movie_id = torch.tensor(movie_id, dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        # Tokenize text features
        tag_text = str(self.tags[idx]) if idx < len(self.tags) else ""
        genre_text = str(self.genres[idx]) if idx < len(self.genres) else ""
        
        tag_enc = self.tokenizer(
            tag_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        genre_enc = self.tokenizer(
            genre_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "tag_input_ids": tag_enc['input_ids'].squeeze(0),
            "tag_attention_mask": tag_enc['attention_mask'].squeeze(0),
            "genre_input_ids": genre_enc['input_ids'].squeeze(0),
            "genre_attention_mask": genre_enc['attention_mask'].squeeze(0),
            "rating": rating
        }

    def get_mappings(self):
        """Return the mappings for use in other datasets"""
        return self.user2idx.copy(), self.movie2idx.copy()


# Enhanced dataset creation function
def create_datasets_safely(df: pd.DataFrame, 
                          usr_features: List[str], 
                          movie_features: List[str], 
                          rating_col: str,
                          tokenizer,
                          test_size: float = 0.2,
                          batch_size: int = 32,
                          max_len: int = 32,
                          random_state: int = 42):
    """
    Create train/test datasets with consistent mappings
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    # Split data first
    train_df, test_df = train_test_split(df, test_size=test_size, 
                                       random_state=random_state, 
                                       stratify=None)  # Can add stratification if needed
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Create training dataset (this creates the mappings)
    train_dataset = CustomDataset(
        train_df, usr_features, movie_features, rating_col,
        tokenizer, max_len=max_len, is_training=True
    )
    
    # Get mappings from training dataset
    user2idx, movie2idx = train_dataset.get_mappings()
    
    # Create test dataset with same mappings
    test_dataset = CustomDataset(
        test_df, usr_features, movie_features, rating_col,
        tokenizer, max_len=max_len, 
        user2idx=user2idx, movie2idx=movie2idx, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, train_df, test_df, user2idx, movie2idx


# Cold start dataset for new users/movies  
class ColdStartDataset(CustomDataset):
    """Dataset that handles completely new users/movies"""
    
    def __init__(self, df: pd.DataFrame, 
                 usr_features: List[str], 
                 movie_features: List[str], 
                 rating_col: str,
                 tokenizer,
                 existing_user2idx: Dict,
                 existing_movie2idx: Dict,
                 max_len: int = 32):
        
        # Add special tokens for unknown entities
        self.UNK_USER_ID = len(existing_user2idx)
        self.UNK_MOVIE_ID = len(existing_movie2idx)
        
        # Extended mappings with UNK tokens
        self.user2idx = existing_user2idx.copy()
        self.user2idx['<UNK_USER>'] = self.UNK_USER_ID
        
        self.movie2idx = existing_movie2idx.copy()  
        self.movie2idx['<UNK_MOVIE>'] = self.UNK_MOVIE_ID
        
        # Call parent init
        super().__init__(df, usr_features, movie_features, rating_col,
                        tokenizer, max_len, self.user2idx, self.movie2idx, 
                        is_training=False)
    
    def __getitem__(self, idx):
        user_raw = self.user_ids_raw[idx]
        movie_raw = self.movie_ids_raw[idx]

        # Map unknown entities to UNK tokens
        user_id = self.user2idx.get(user_raw, self.UNK_USER_ID)
        movie_id = self.movie2idx.get(movie_raw, self.UNK_MOVIE_ID)

        # Rest is same as parent
        user_id = torch.tensor(user_id, dtype=torch.long)
        movie_id = torch.tensor(movie_id, dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        tag_text = str(self.tags[idx]) if idx < len(self.tags) else ""
        genre_text = str(self.genres[idx]) if idx < len(self.genres) else ""
        
        tag_enc = self.tokenizer(tag_text, padding='max_length', truncation=True,
                               max_length=self.max_len, return_tensors='pt')
        genre_enc = self.tokenizer(genre_text, padding='max_length', truncation=True,
                                 max_length=self.max_len, return_tensors='pt')

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "tag_input_ids": tag_enc['input_ids'].squeeze(0),
            "tag_attention_mask": tag_enc['attention_mask'].squeeze(0),
            "genre_input_ids": genre_enc['input_ids'].squeeze(0),
            "genre_attention_mask": genre_enc['attention_mask'].squeeze(0),
            "rating": rating
        }