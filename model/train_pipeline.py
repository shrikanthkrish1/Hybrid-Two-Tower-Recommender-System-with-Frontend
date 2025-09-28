# model/train_pipeline.py - FIXED VERSION with MongoDB integration
import os
import torch
from transformers import AutoTokenizer
from two_tower import TwoTower
from dataset import CustomDataset, create_datasets_safely
from mongo_data_loader import MongoDataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from datetime import datetime
import json
import logging

class MovieRecommendationSystem:
    def __init__(self, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = None
        self.xgb_model = None
        self.user2idx = None
        self.movie2idx = None
        self.idx2user = None
        self.idx2movie = None
        self.df = None
        self.mongo_loader = MongoDataLoader()
        self.training_metadata = {}

    def load_data_from_mongodb(self, 
                              days_back: int = None, 
                              min_ratings_per_user: int = 5,
                              min_ratings_per_movie: int = 5):
        """Load training data directly from MongoDB"""
        logging.info("Loading data from MongoDB...")
        
        # Get data statistics
        stats = self.mongo_loader.get_data_stats()
        logging.info(f"Data stats: {stats}")
        
        # Fetch training data
        self.df = self.mongo_loader.fetch_training_data(
            days_back=days_back,
            min_ratings_per_user=min_ratings_per_user,
            min_ratings_per_movie=min_ratings_per_movie
        )
        
        if self.df.empty:
            raise ValueError("No data loaded from MongoDB")
        
        # Store metadata
        self.training_metadata = {
            "training_date": datetime.now().isoformat(),
            "total_ratings": len(self.df),
            "unique_users": self.df['userId'].nunique(),
            "unique_movies": self.df['movieId'].nunique(),
            "data_date_range": {
                "start": self.df['timestamp'].min() if 'timestamp' in self.df.columns else None,
                "end": self.df['timestamp'].max() if 'timestamp' in self.df.columns else None
            },
            "filtering_params": {
                "days_back": days_back,
                "min_ratings_per_user": min_ratings_per_user,
                "min_ratings_per_movie": min_ratings_per_movie
            }
        }
        
        logging.info(f"Loaded {len(self.df)} ratings from MongoDB")
        return self.df

    def should_retrain(self, threshold_new_ratings: int = 1000,
                      threshold_days: int = 7) -> dict:
        """
        Determine if model should be retrained based on new data
        """
        # Check if model exists
        model_path = os.environ.get("MODEL_ARTIFACT", "artifacts/model.pt")
        if not os.path.exists(model_path):
            return {"should_retrain": True, "reason": "No existing model found"}
        
        # Load last training metadata
        metadata_path = "artifacts/training_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                last_metadata = json.load(f)
                last_training_date = datetime.fromisoformat(last_metadata["training_date"])
        else:
            return {"should_retrain": True, "reason": "No training metadata found"}
        
        # Check time since last training
        days_since_training = (datetime.now() - last_training_date).days
        if days_since_training >= threshold_days:
            return {"should_retrain": True, "reason": f"Training is {days_since_training} days old"}
        
        # Check new data volume
        new_data = self.mongo_loader.get_new_data_since_last_training(last_training_date)
        new_ratings_count = len(new_data)
        
        if new_ratings_count >= threshold_new_ratings:
            return {
                "should_retrain": True, 
                "reason": f"Found {new_ratings_count} new ratings since last training"
            }
        
        return {
            "should_retrain": False, 
            "reason": f"Only {new_ratings_count} new ratings in {days_since_training} days"
        }

    def load_data(self, movies_path="data/movies.csv", ratings_path="data/ratings.csv", tags_path="data/tags.csv"):
        """Legacy method - loads from CSV files"""
        try:
            movies_data = pd.read_csv(movies_path)
            ratings_data = pd.read_csv(ratings_path)
            
            try:
                tags_data = pd.read_csv(tags_path)
            except Exception:
                tags_data = pd.DataFrame(columns=['userId','movieId','tag'])
                
            # merge: left-join ratings <- tags on userId,movieId then merge movies
            df = ratings_data.merge(tags_data, on=['userId','movieId'], how='left')
            df = df.merge(movies_data, on='movieId', how='left')
            df['tag'] = df.get('tag', pd.Series([""]*len(df))).fillna("")
            df['genres'] = df.get('genres', pd.Series([""]*len(df))).fillna("")
            df['title'] = df.get('title', pd.Series([f"Movie_{i}" for i in df['movieId']])).fillna("")
            
            self.df = df
            return df
        except Exception as e:
            logging.error(f"Error loading CSV data: {e}")
            raise

    def create_datasets(self, test_size=0.2, batch_size=32):
        """Create datasets using FIXED dataset creation method"""
        if self.df is None:
            raise RuntimeError("Load data first")
        
        # Use the SAFE dataset creation function to avoid mapping conflicts
        train_loader, test_loader, train_df, test_df, user2idx, movie2idx = create_datasets_safely(
            self.df, 
            ['userId', 'tag'], 
            ['movieId', 'genres'], 
            'rating',
            self.tokenizer,
            test_size=test_size,
            batch_size=batch_size
        )
        
        # Store mappings
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.idx2user = {i: u for u, i in user2idx.items()}
        self.idx2movie = {i: m for m, i in movie2idx.items()}
        
        return train_loader, test_loader, train_df, test_df

    def create_model(self):
        """Create model with proper configuration"""
        if self.user2idx is None or self.movie2idx is None:
            raise RuntimeError("Create datasets first to get proper mappings")
            
        config = {
            "user_embedding_num": len(self.user2idx),
            "user_embedding_dim": 64,
            "item_embedding_num": len(self.movie2idx),
            "item_embedding_dim": 64,
            "user_dense": [64, 32],
            "item_dense": [64, 32],
            "activation": "relu",
        }
        self.model = TwoTower(config, 
                             vocab_size_tag=self.tokenizer.vocab_size, 
                             vocab_size_genre=self.tokenizer.vocab_size)
        return self.model

    def train(self, train_loader, test_loader, epochs=3, lr=1e-3, device="cpu"):
        """Training with better error handling"""
        if self.model is None:
            self.create_model()
            
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for ep in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0
            n = 0
            
            for batch in train_loader:
                try:
                    optimizer.zero_grad()
                    preds = self.model(batch['user_id'].to(device), batch['tag_input_ids'].to(device),
                                       batch['movie_id'].to(device), batch['genre_input_ids'].to(device))
                    
                    targets = batch['rating'].to(device)
                    
                    # Handle shape mismatches
                    if preds.shape != targets.shape:
                        if preds.dim() == 0:
                            preds = preds.unsqueeze(0)
                        if targets.dim() == 0:
                            targets = targets.unsqueeze(0)
                    
                    loss = loss_fn(preds, targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item() * batch['rating'].size(0)
                    n += batch['rating'].size(0)
                except Exception as e:
                    logging.warning(f"Error in training batch: {e}")
                    continue
            
            avg_train_loss = total_loss / max(1, n)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            total_eval_loss = 0.0
            n_eval = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        preds = self.model(batch['user_id'].to(device), batch['tag_input_ids'].to(device),
                                           batch['movie_id'].to(device), batch['genre_input_ids'].to(device))
                        targets = batch['rating'].to(device)
                        
                        # Handle shape mismatches
                        if preds.shape != targets.shape:
                            if preds.dim() == 0:
                                preds = preds.unsqueeze(0)
                            if targets.dim() == 0:
                                targets = targets.unsqueeze(0)
                        
                        loss = loss_fn(preds, targets)
                        total_eval_loss += loss.item() * batch['rating'].size(0)
                        n_eval += batch['rating'].size(0)
                    except Exception as e:
                        logging.warning(f"Error in validation batch: {e}")
                        continue
            
            avg_val_loss = total_eval_loss / max(1, n_eval)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {ep+1}/{epochs} Train loss: {avg_train_loss:.4f} Val loss: {avg_val_loss:.4f}")
        
        # Store training history
        self.training_metadata.update({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None
        })
        
        return self.model

    def extract_features(self, loader, device="cpu"):
        """Extract features for XGBoost training"""
        self.model.eval()
        Xs = []
        Ys = []
        
        with torch.no_grad():
            for batch in loader:
                try:
                    user_vec = self.model.user_embedding_vec(batch['user_id'].to(device), 
                                                           batch['tag_input_ids'].to(device)).cpu().numpy()
                    item_vec = self.model.item_embedding_vec(batch['movie_id'].to(device), 
                                                           batch['genre_input_ids'].to(device)).cpu().numpy()
                    y = batch['rating'].cpu().numpy()
                    
                    # ensure shapes
                    if user_vec.ndim == 1:
                        user_vec = user_vec.reshape(1, -1)
                    if item_vec.ndim == 1:
                        item_vec = item_vec.reshape(1, -1)
                        
                    Xs.append(np.hstack([user_vec, item_vec]))
                    Ys.append(y)
                except Exception as e:
                    logging.warning(f"Error extracting features: {e}")
                    continue
        
        if not Xs:
            raise ValueError("No features extracted successfully")
            
        X = np.vstack(Xs)
        y = np.concatenate(Ys)
        return X, y

    def train_xgb(self, train_loader, test_loader, device="cpu"):
        """Train XGBoost model"""
        X_train, y_train = self.extract_features(train_loader, device)
        X_test, y_test = self.extract_features(test_loader, device)
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, 
                           random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        self.xgb_model = model
        
        # Calculate performance metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.training_metadata.update({
            "xgb_train_r2": train_score,
            "xgb_test_r2": test_score
        })
        
        print(f"XGBoost Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        return model

    def save_artifact(self, path="artifacts/model.pt"):
        """Save model with comprehensive metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Update metadata
        self.training_metadata.update({
            "model_save_time": datetime.now().isoformat(),
            "total_users": len(self.user2idx) if self.user2idx else 0,
            "total_movies": len(self.movie2idx) if self.movie2idx else 0
        })
        
        save_dict = {
            "model_state": self.model.state_dict() if self.model else None,
            "xgb_model": self.xgb_model,
            "user2idx": self.user2idx,
            "movie2idx": self.movie2idx,
            "idx2user": self.idx2user,
            "idx2movie": self.idx2movie,
            "training_metadata": self.training_metadata
        }
        
        torch.save(save_dict, path)
        
        # Save metadata separately for easy access
        metadata_path = os.path.join(os.path.dirname(path), "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        logging.info(f"Model and metadata saved to {path}")
        return path

    def load_artifact(self, path="artifacts/model.pt"):
        """FIXED: Complete model loading implementation"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found at {path}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Load mappings
            self.user2idx = checkpoint.get("user2idx", {})
            self.movie2idx = checkpoint.get("movie2idx", {})
            self.idx2user = checkpoint.get("idx2user", {})
            self.idx2movie = checkpoint.get("idx2movie", {})
            self.training_metadata = checkpoint.get("training_metadata", {})
            self.xgb_model = checkpoint.get("xgb_model")
            
            # Recreate model architecture if state exists
            if checkpoint.get("model_state") and self.user2idx and self.movie2idx:
                config = {
                    "user_embedding_num": len(self.user2idx),
                    "user_embedding_dim": 64,
                    "item_embedding_num": len(self.movie2idx),
                    "item_embedding_dim": 64,
                    "user_dense": [64, 32],
                    "item_dense": [64, 32],
                    "activation": "relu",
                }
                self.model = TwoTower(config, 
                                    vocab_size_tag=self.tokenizer.vocab_size, 
                                    vocab_size_genre=self.tokenizer.vocab_size)
                self.model.load_state_dict(checkpoint["model_state"])
                self.model.eval()
            
            logging.info(f"Model loaded from {path}")
            logging.info(f"Users: {len(self.user2idx)}, Movies: {len(self.movie2idx)}")
            
            return self
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def recommend_for_user(self, user_id: str, top_k: int = 10) -> list[dict]:
        """FIXED: Generate ML-based recommendations for a user"""
        if not self.model or not self.xgb_model:
            logging.warning("Model not loaded, cannot generate recommendations")
            return []
        
        # Check if user exists in training data
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            # Handle new user - return popular movies
            return self._get_popular_movies(top_k)
        
        # Get all movies for recommendation
        all_movie_ids = list(self.movie2idx.keys())
        if not all_movie_ids:
            return []
        
        # Create prediction data
        predictions = []
        batch_size = 32
        
        self.model.eval()
        device = next(self.model.parameters()).device if self.model else torch.device('cpu')
        
        with torch.no_grad():
            for i in range(0, len(all_movie_ids), batch_size):
                batch_movies = all_movie_ids[i:i + batch_size]
                
                # Create batch data
                user_ids = [user_id] * len(batch_movies)
                
                # Create dummy DataFrame for feature extraction
                batch_df = pd.DataFrame({
                    'userId': user_ids,
                    'movieId': batch_movies,
                    'rating': [0.0] * len(batch_movies),  # Dummy ratings
                    'tag': [''] * len(batch_movies),
                    'genres': [''] * len(batch_movies)
                })
                
                try:
                    # Create dataset for batch prediction
                    from dataset import CustomDataset
                    batch_dataset = CustomDataset(
                        batch_df, ['userId', 'tag'], ['movieId', 'genres'], 'rating',
                        self.tokenizer, user2idx=self.user2idx, movie2idx=self.movie2idx,
                        is_training=False
                    )
                    
                    batch_loader = DataLoader(batch_dataset, batch_size=len(batch_movies), shuffle=False)
                    
                    # Extract features and predict
                    for batch in batch_loader:
                        user_vec = self.model.user_embedding_vec(
                            batch['user_id'].to(device), 
                            batch['tag_input_ids'].to(device)
                        ).cpu().numpy()
                        
                        item_vec = self.model.item_embedding_vec(
                            batch['movie_id'].to(device), 
                            batch['genre_input_ids'].to(device)
                        ).cpu().numpy()
                        
                        # Combine features for XGBoost
                        if user_vec.ndim == 1:
                            user_vec = user_vec.reshape(1, -1)
                        if item_vec.ndim == 1:
                            item_vec = item_vec.reshape(1, -1)
                            
                        features = np.hstack([user_vec, item_vec])
                        scores = self.xgb_model.predict(features)
                        
                        # Store predictions
                        for j, (movie_id, score) in enumerate(zip(batch_movies, scores)):
                            predictions.append({
                                "movie_id": str(movie_id),
                                "score": float(score),
                                "title": f"Movie_{movie_id}"
                            })
                        break  # Only one batch per iteration
                        
                except Exception as e:
                    logging.warning(f"Error predicting batch: {e}")
                    continue
        
        # Sort by score and return top-k
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, pred in enumerate(predictions[:top_k]):
            pred['rank'] = i + 1
        
        return predictions[:top_k]
    
    def _get_popular_movies(self, top_k: int = 10) -> list[dict]:
        """Fallback recommendations for new users"""
        if not hasattr(self, 'df') or self.df is None:
            return []
        
        try:
            # Calculate popularity based on available data
            movie_stats = self.df.groupby('movieId').agg({
                'rating': ['mean', 'count', 'std']
            }).reset_index()
            
            movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'rating_std']
            movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
            
            # Popularity score: weighted average with count and low variance bonus
            movie_stats['popularity_score'] = (
                movie_stats['avg_rating'] * 
                np.log(1 + movie_stats['rating_count']) * 
                (1 + 1/(1 + movie_stats['rating_std']))  # Lower std = higher score
            )
            
            # Get top movies
            top_movies = movie_stats.nlargest(top_k, 'popularity_score')
            
            return [
                {
                    "movie_id": str(row['movieId']),
                    "score": float(row['popularity_score']),
                    "title": f"Movie_{row['movieId']}",
                    "rank": i + 1
                }
                for i, (_, row) in enumerate(top_movies.iterrows())
            ]
            
        except Exception as e:
            logging.error(f"Error getting popular movies: {e}")
            return []