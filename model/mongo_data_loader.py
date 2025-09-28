# model/mongodb_data_loader.py
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List
import logging

class MongoDataLoader:
    def __init__(self, mongo_uri: str = None):
        self.mongo_uri = mongo_uri or os.environ.get(
            "MONGO_URI",
            "mongodb+srv://shrikanthkrish1_db_user:GZpOkj3oSBZkdGRQ@cluster0.axumhzh.mongodb.net/?retryWrites=true&w=majority"
        )
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["movie_recommender"]
        self.ratings_collection = self.db["ratings"]
        logging.info("✅ MongoDB Data Loader initialized")
        
    def fetch_training_data(self, 
                          days_back: Optional[int] = None,
                          min_ratings_per_user: int = 5,
                          min_ratings_per_movie: int = 5) -> pd.DataFrame:
        """
        Fetch training data from MongoDB with filtering
        """
        # Build query
        query = {}
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query["timestamp"] = {"$gte": cutoff_date.timestamp()}
        
        logging.info(f"Fetching data with query: {query}")
        
        # Fetch all ratings
        cursor = self.ratings_collection.find(query)
        ratings_data = list(cursor)
        
        if not ratings_data:
            logging.warning("No rating data found in MongoDB")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(ratings_data)
        
        # Data cleaning and transformation
        df['userId'] = df['user_id']
        df['movieId'] = df['movie_id']
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['tag'] = df.get('tag', '').fillna('')
        df['genres'] = df.get('genres', '').fillna('')
        
        # Add movie titles (basic implementation)
        df['title'] = df['movieId'].apply(lambda x: f"Movie_{x}")
        
        # Remove invalid ratings
        df = df.dropna(subset=['rating'])
        df = df[df['rating'].between(0.5, 5.0)]
        
        # Filter users and movies with minimum interactions
        user_counts = df['userId'].value_counts()
        movie_counts = df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
        
        initial_len = len(df)
        df = df[df['userId'].isin(valid_users) & df['movieId'].isin(valid_movies)]
        
        logging.info(f"Loaded {len(df)} ratings (filtered from {initial_len}) for {df['userId'].nunique()} users and {df['movieId'].nunique()} movies")
        
        return df
    
    def get_data_stats(self) -> Dict:
        """Get statistics about the data in MongoDB"""
        try:
            total_ratings = self.ratings_collection.count_documents({})
            
            # Get date range
            oldest = self.ratings_collection.find_one(sort=[("timestamp", 1)])
            newest = self.ratings_collection.find_one(sort=[("timestamp", -1)])
            
            # Get unique counts using aggregation
            pipeline = [
                {"$group": {
                    "_id": None,
                    "unique_users": {"$addToSet": "$user_id"},
                    "unique_movies": {"$addToSet": "$movie_id"},
                    "avg_rating": {"$avg": "$rating"}
                }},
                {"$project": {
                    "unique_users": {"$size": "$unique_users"},
                    "unique_movies": {"$size": "$unique_movies"},
                    "avg_rating": 1
                }}
            ]
            
            stats_result = list(self.ratings_collection.aggregate(pipeline))
            stats = stats_result[0] if stats_result else {}
            
            return {
                "total_ratings": total_ratings,
                "unique_users": stats.get("unique_users", 0),
                "unique_movies": stats.get("unique_movies", 0),
                "avg_rating": stats.get("avg_rating", 0),
                "date_range": {
                    "oldest": datetime.fromtimestamp(oldest.get("timestamp")) if oldest and "timestamp" in oldest else None,
                    "newest": datetime.fromtimestamp(newest.get("timestamp")) if newest and "timestamp" in newest else None
                }
            }
        except Exception as e:
            logging.error(f"Error getting data stats: {e}")
            return {"error": str(e)}
    
    def get_new_data_since_last_training(self, last_training_date: datetime) -> pd.DataFrame:
        """Get only new data since last training"""
        query = {"timestamp": {"$gt": last_training_date.timestamp()}}
        return self.fetch_training_data_with_query(query)
        
    def fetch_training_data_with_query(self, query: Dict) -> pd.DataFrame:
        """Helper method to fetch data with custom query"""
        try:
            cursor = self.ratings_collection.find(query)
            ratings_data = list(cursor)
            
            if not ratings_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(ratings_data)
            df['userId'] = df['user_id']
            df['movieId'] = df['movie_id']
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['tag'] = df.get('tag', '').fillna('')
            df['genres'] = df.get('genres', '').fillna('')
            df['title'] = df['movieId'].apply(lambda x: f"Movie_{x}")
            
            # Remove invalid ratings
            df = df.dropna(subset=['rating'])
            df = df[df['rating'].between(0.5, 5.0)]
            
            return df
        except Exception as e:
            logging.error(f"Error fetching data with query {query}: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")