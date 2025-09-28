from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import redis
import json
import numpy as np
import os
import time
from transformers import AutoTokenizer
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the model
import sys
sys.path.append('.')
from model.two_tower import TwoTower

app = FastAPI(title="TwoTower Movie Recommender API")

# Configuration
print(f"DEBUG: MONGO_URI loaded = {os.environ.get('MONGO_URI', 'NOT FOUND')}")
print(f"DEBUG: Current working directory = {os.getcwd()}")
print(f"DEBUG: .env file exists = {os.path.exists('.env')}")

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pt")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")  # Add default fallback
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"DEBUG: Final MONGO_URI = {MONGO_URI}")

# Global variables
redis_client = redis.from_url(REDIS_URL)
model = None
tokenizer = None
user2idx = {}
movie2idx = {}
idx2user = {}
idx2movie = {}
item_embeddings = None

class RatingInput(BaseModel):
    user_id: str
    movie_id: str
    rating: float
    comment: str = ""
    tag: str = ""
    genre: str = ""

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict]
    model_type: str = "TwoTower"

class UserPreferences(BaseModel):
    user_id: str
    preferred_genres: List[str]
    disliked_genres: List[str] = []
    preference_strength: float = 1.0

def get_mongo_client():
    """Get MongoDB client with proper error handling"""
    try:
        from pymongo import MongoClient
        
        if MONGO_URI is None:
            print("MongoDB URI is None - .env file not loaded properly")
            return None
        
        print(f"Attempting MongoDB connection...")
        print(f"Using URI: {MONGO_URI[:50]}...")  # Don't log full URI with password
        
        if "mongodb+srv" in MONGO_URI:
            # Atlas connection
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000, retryWrites=True)
        else:
            # Local MongoDB connection
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            
        # Test connection
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return client
        
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def load_model():
    """Load the trained TwoTower model"""
    global model, tokenizer, user2idx, movie2idx, idx2user, idx2movie, item_embeddings
    
    try:
        print("Loading TwoTower model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load model checkpoint
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            user2idx = checkpoint["user2idx"]
            movie2idx = checkpoint["movie2idx"]
            idx2user = {v: k for k, v in user2idx.items()}
            idx2movie = {v: k for k, v in movie2idx.items()}
            
            # Create model config
            config = {
                "user_embedding_num": len(user2idx),
                "user_embedding_dim": 64,
                "item_embedding_num": len(movie2idx),
                "item_embedding_dim": 64,
                "user_dense": [128, 64, 32],
                "item_dense": [128, 64, 32],
                "activation": "relu",
            }
            
            # Initialize model
            model = TwoTower(config, tokenizer.vocab_size, tokenizer.vocab_size)
            model.load_state_dict(checkpoint["model_state"])
            model.to(DEVICE)
            model.eval()
            
            # Precompute item embeddings for fast inference
            precompute_item_embeddings()
            
            print(f"Model loaded successfully! Users: {len(user2idx)}, Movies: {len(movie2idx)}")
            
        else:
            print(f"Model file not found at {MODEL_PATH}, creating mock model...")
            create_mock_model()
            
    except Exception as e:
        print(f"Error loading model: {e}")
        create_mock_model()

def create_mock_model():
    """Create a mock model for testing when trained model isn't available"""
    global model, tokenizer, user2idx, movie2idx, idx2user, idx2movie, item_embeddings
    
    # Create mock data
    mock_users = [f"user_{i}" for i in range(100)]
    mock_movies = [f"movie_{i}" for i in range(1000)]
    
    user2idx = {u: i for i, u in enumerate(mock_users)}
    movie2idx = {m: i for i, m in enumerate(mock_movies)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    config = {
        "user_embedding_num": len(user2idx),
        "user_embedding_dim": 64,
        "item_embedding_num": len(movie2idx),
        "item_embedding_dim": 64,
        "user_dense": [128, 64, 32],
        "item_dense": [128, 64, 32],
        "activation": "relu",
    }
    
    model = TwoTower(config, tokenizer.vocab_size, tokenizer.vocab_size)
    model.to(DEVICE)
    model.eval()
    
    # Initialize with random weights
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    
    precompute_item_embeddings()
    print("Mock model created for testing")

def precompute_item_embeddings():
    """Precompute all item embeddings for fast recommendation generation"""
    global item_embeddings
    
    try:
        with torch.no_grad():
            n_items = len(movie2idx)
            item_ids = torch.arange(n_items, dtype=torch.long, device=DEVICE)
            
            # Use empty genre tokens (in production, you'd use actual genre data)
            genre_tokens = torch.zeros((n_items, 1), dtype=torch.long, device=DEVICE)
            
            item_embeddings = model.item_embedding_vec(item_ids, genre_tokens)
            item_embeddings = item_embeddings.cpu().numpy()
            
            print(f"Precomputed {n_items} item embeddings")
            
    except Exception as e:
        print(f"Error precomputing item embeddings: {e}")
        item_embeddings = np.random.randn(len(movie2idx), 32)

def store_rating_in_mongo(rating_data: dict):
    """Store rating in MongoDB for future training"""
    try:
        client = get_mongo_client()
        if client is None:
            print("MongoDB not available, skipping rating storage")
            return False
            
        db = client["movie_recommender"]
        collection = db["ratings"]
        
        # Add timestamp
        rating_data["timestamp"] = time.time()
        
        # Insert into MongoDB
        collection.insert_one(rating_data)
        print(f"Stored rating in MongoDB: {rating_data['user_id']} -> {rating_data['movie_id']}")
        client.close()
        return True
        
    except Exception as e:
        print(f"Could not store rating in MongoDB: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()
    
    # Test MongoDB connection during startup
    print("Testing MongoDB Atlas connection...")
    mongo_client = get_mongo_client()
    if mongo_client:
        print("✅ MongoDB Atlas connection successful!")
        mongo_client.close()
    else:
        print("❌ MongoDB Atlas connection failed - preferences will not work")
    
    print("API server started successfully")

@app.post("/submit_rating", response_model=dict)
async def submit_rating(rating: RatingInput):
    """Submit a rating and get updated recommendations"""
    
    # Store rating data
    rating_data = {
        "user_id": rating.user_id,
        "movie_id": rating.movie_id,
        "rating": rating.rating,
        "comment": rating.comment,
        "tag": rating.tag,
        "genre": rating.genre,
        "timestamp": time.time()
    }
    
    # Store in MongoDB for training
    stored_in_mongo = store_rating_in_mongo(rating_data)
    
    # Try to queue for batch processing
    try:
        redis_client.rpush("rating_queue", json.dumps(rating_data))
        redis_client.delete(f"rec:{rating.user_id}")
    except Exception as e:
        print(f"Redis not available: {e}")
    
    # Generate new recommendations immediately
    recommendations = await generate_recommendations(rating.user_id)
    
    # Try to cache the recommendations
    try:
        redis_client.setex(f"rec:{rating.user_id}", 3600, json.dumps(recommendations))
    except Exception as e:
        print(f"Could not cache recommendations: {e}")
    
    return {
        "status": "success",
        "message": f"Rating submitted and {len(recommendations)} recommendations generated",
        "recommendations_preview": recommendations[:3],
        "stored_in_database": stored_in_mongo
    }

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, top_k: int = 10):
    """Get recommendations for a user"""
    
    # Try to check cache first
    try:
        cached = redis_client.get(f"rec:{user_id}")
        if cached:
            recommendations = json.loads(cached)[:top_k]
            return RecommendationResponse(user_id=user_id, recommendations=recommendations)
    except Exception as e:
        print(f"Cache unavailable: {e}")
    
    # Generate new recommendations
    recommendations = await generate_recommendations(user_id, top_k)
    
    # Try to cache for 1 hour
    try:
        redis_client.setex(f"rec:{user_id}", 3600, json.dumps(recommendations))
    except Exception as e:
        print(f"Could not cache recommendations: {e}")
    
    return RecommendationResponse(user_id=user_id, recommendations=recommendations)

async def generate_recommendations(user_id: str, top_k: int = 10):
    """Generate recommendations using TwoTower model"""
    
    if model is None or item_embeddings is None:
        return []
    
    try:
        # Handle new users not in training data
        if user_id not in user2idx:
            # Add new user to mappings
            new_user_idx = len(user2idx)
            user2idx[user_id] = new_user_idx
            idx2user[new_user_idx] = user_id
            print(f"Added new user: {user_id} -> {new_user_idx}")
            
            # For new users, try preference-based recommendations first
            return await generate_cold_start_recommendations_with_preferences(user_id, top_k)
        
        user_idx = user2idx[user_id]
        
        with torch.no_grad():
            # Get user embedding
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=DEVICE)
            # Use empty tag tokens (in production, use actual user tags)
            tag_tokens = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            
            user_embedding = model.user_embedding_vec(user_tensor, tag_tokens)
            user_embedding = user_embedding.cpu().numpy().reshape(-1)
            
            # Compute similarities with all items
            similarities = np.dot(item_embeddings, user_embedding)
            
            # Get top-k items
            top_indices = np.argsort(-similarities)[:top_k]
            
            recommendations = []
            for rank, idx in enumerate(top_indices):
                movie_id = idx2movie.get(idx, f"movie_{idx}")
                score = float(similarities[idx])
                
                recommendations.append({
                    "movie_id": movie_id,
                    "predicted_rating": min(5.0, max(1.0, score * 2.5 + 2.5)),  # Scale to 1-5
                    "confidence": min(1.0, abs(score)),
                    "score": score,
                    "rank": rank + 1,
                    "title": f"Movie {movie_id}",
                    "method": "TwoTower Neural Collaborative Filtering"
                })
            
            return recommendations
            
    except Exception as e:
        print(f"Error generating recommendations for {user_id}: {e}")
        return await generate_cold_start_recommendations_with_preferences(user_id, top_k)

def generate_cold_start_recommendations(top_k: int = 10):
    """Generate basic cold start recommendations"""
    
    # Return random popular movies for cold start
    if len(movie2idx) == 0:
        return []
        
    popular_movie_indices = np.random.choice(len(movie2idx), 
                                           size=min(top_k, len(movie2idx)), 
                                           replace=False)
    
    recommendations = []
    for rank, idx in enumerate(popular_movie_indices):
        movie_id = idx2movie.get(idx, f"movie_{idx}")
        
        recommendations.append({
            "movie_id": movie_id,
            "predicted_rating": np.random.uniform(3.5, 4.5),
            "confidence": 0.5,
            "score": np.random.uniform(0.3, 0.7),
            "rank": rank + 1,
            "title": f"Movie {movie_id}",
            "method": "Cold Start (Popular Items)"
        })
    
    return recommendations

async def generate_cold_start_recommendations_with_preferences(user_id: str, top_k: int = 10):
    """Generate recommendations for new users using their genre preferences"""
    
    try:
        client = get_mongo_client()
        if client is None:
            print("MongoDB not available for preferences, using fallback")
            return generate_cold_start_recommendations(top_k)
            
        db = client["movie_recommender"]
        collection = db["user_preferences"]
        
        prefs = collection.find_one({"user_id": user_id})
        client.close()
        
        if prefs and prefs.get("preferred_genres"):
            # Generate recommendations based on preferred genres
            preferred_genres = prefs["preferred_genres"]
            disliked_genres = prefs.get("disliked_genres", [])
            
            recommendations = []
            
            # Create genre-based scoring
            movie_ids = list(idx2movie.keys())
            for i, movie_idx in enumerate(movie_ids):
                if i >= top_k * 3:  # Limit search space
                    break
                    
                movie_id_str = str(idx2movie[movie_idx])
                
                # Simple genre matching score
                # In production, you'd have actual genre data for movies
                genre_score = 0.0
                
                # Mock genre assignment based on movie_id for demo
                mock_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Adventure", "Horror"]
                # Use modulo to assign genres consistently
                try:
                    movie_id_num = int(movie_id_str) if movie_id_str.isdigit() else hash(movie_id_str)
                    mock_movie_genres = [mock_genres[movie_id_num % len(mock_genres)]]
                except:
                    mock_movie_genres = ["Drama"]  # Default fallback
                
                # Score based on preference match
                for genre in mock_movie_genres:
                    if genre in preferred_genres:
                        genre_score += 1.0
                    elif genre in disliked_genres:
                        genre_score -= 0.5
                
                if genre_score > 0:
                    recommendations.append({
                        "movie_id": movie_id_str,
                        "predicted_rating": min(5.0, 3.5 + (genre_score * 0.5)),
                        "confidence": 0.7,
                        "score": genre_score,
                        "rank": 0,  # Will be set later
                        "title": f"Movie {movie_id_str}",
                        "method": "Genre-Based Cold Start",
                        "genres": mock_movie_genres
                    })
            
            # Sort by genre score
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            
            # Add ranks
            for i, rec in enumerate(recommendations[:top_k]):
                rec["rank"] = i + 1
            
            print(f"Generated {len(recommendations[:top_k])} preference-based recommendations for {user_id}")
            return recommendations[:top_k]
            
    except Exception as e:
        print(f"Error in preference-based recommendations: {e}")
    
    # Fallback to random recommendations
    return generate_cold_start_recommendations(top_k)

@app.post("/user/preferences")
async def save_user_preferences(preferences: UserPreferences):
    """Save user genre preferences to MongoDB permanently"""
    try:
        print(f"Attempting to save preferences for user: {preferences.user_id}")
        print(f"Preferred genres: {preferences.preferred_genres}")
        
        client = get_mongo_client()
        if client is None:
            print("MongoDB connection failed")
            raise HTTPException(status_code=503, detail="Database unavailable - check MongoDB connection")
        
        db = client["movie_recommender"]
        collection = db["user_preferences"]
        
        preference_data = {
            "user_id": preferences.user_id,
            "preferred_genres": preferences.preferred_genres,
            "disliked_genres": preferences.disliked_genres,
            "preference_strength": preferences.preference_strength,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Replace existing preferences or create new
        result = collection.replace_one(
            {"user_id": preferences.user_id}, 
            preference_data, 
            upsert=True
        )
        
        client.close()
        
        print(f"✅ Successfully saved preferences for user {preferences.user_id}")
        print(f"MongoDB result - Modified: {result.modified_count}, Upserted: {result.upserted_id}")
        
        # Clear cached recommendations to force regeneration with new preferences
        try:
            redis_client.delete(f"rec:{preferences.user_id}")
        except Exception as e:
            print(f"Could not clear cache: {e}")
        
        return {"status": "success", "message": "Preferences saved permanently"}
        
    except Exception as e:
        print(f"❌ Error saving preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save preferences: {str(e)}")

@app.get("/user/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get user preferences from MongoDB"""
    try:
        print(f"Getting preferences for user: {user_id}")
        
        client = get_mongo_client()
        if client is None:
            print("MongoDB connection failed for getting preferences")
            return {"has_preferences": False, "error": "Database unavailable"}
        
        db = client["movie_recommender"]
        collection = db["user_preferences"]
        
        prefs = collection.find_one({"user_id": user_id})
        client.close()
        
        if prefs:
            result = {
                "has_preferences": True,
                "preferred_genres": prefs.get("preferred_genres", []),
                "disliked_genres": prefs.get("disliked_genres", []),
                "created_at": prefs.get("created_at"),
                "updated_at": prefs.get("updated_at")
            }
            print(f"Found preferences for {user_id}: {result}")
            return result
        else:
            print(f"No preferences found for user: {user_id}")
            return {"has_preferences": False}
            
    except Exception as e:
        print(f"Error getting preferences: {e}")
        return {"has_preferences": False, "error": str(e)}

@app.get("/debug/test-preferences")
async def test_preferences():
    """Debug endpoint to test MongoDB connection and preferences functionality"""
    try:
        client = get_mongo_client()
        if client is None:
            return {"error": "MongoDB connection failed"}
        
        db = client["movie_recommender"]
        collection = db["user_preferences"]
        
        # Test write
        test_data = {
            "user_id": "test_user_debug_123",
            "preferred_genres": ["Action", "Comedy"],
            "disliked_genres": ["Horror"],
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        collection.replace_one({"user_id": "test_user_debug_123"}, test_data, upsert=True)
        
        # Test read
        result = collection.find_one({"user_id": "test_user_debug_123"})
        
        # Count total documents
        total_prefs = collection.count_documents({})
        
        client.close()
        
        return {
            "mongodb_working": True,
            "test_write_successful": True,
            "test_read_successful": result is not None,
            "total_preferences_in_db": total_prefs,
            "test_data": result,
            "mongo_uri_partial": MONGO_URI[:50] + "..."
        }
        
    except Exception as e:
        return {
            "mongodb_working": False,
            "error": str(e),
            "mongo_uri_partial": MONGO_URI[:50] + "..."
        }

@app.get("/model/status")
async def model_status():
    """Get model status and statistics"""
    return {
        "model_loaded": model is not None,
        "model_type": "TwoTower",
        "device": str(DEVICE),
        "total_users": len(user2idx),
        "total_movies": len(movie2idx),
        "embeddings_precomputed": item_embeddings is not None,
        "embedding_dim": item_embeddings.shape[1] if item_embeddings is not None else None
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the model from disk"""
    load_model()
    return {"status": "Model reloaded successfully"}

@app.get("/health")
async def health_check():
    redis_connected = False
    try:
        redis_connected = redis_client.ping()
    except:
        pass
    
    mongo_connected = get_mongo_client() is not None
        
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "redis_connected": redis_connected,
        "mongo_connected": mongo_connected,
        "total_users": len(user2idx),
        "total_movies": len(movie2idx)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)