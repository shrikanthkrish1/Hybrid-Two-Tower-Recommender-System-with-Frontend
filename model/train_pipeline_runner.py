import os
import sys
import logging
from datetime import datetime

# Fix imports to match your file structure
sys.path.append('.')
from model.train_pipeline import MovieRecommendationSystem
from model.mongo_data_loader import MongoDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def should_train_model():
    """
    Intelligent training decision based on:
    1. Model existence
    2. New data availability
    3. Time since last training
    """
    mr = MovieRecommendationSystem()
    
    try:
        # Check if we should retrain
        retrain_decision = mr.should_retrain(
            threshold_new_ratings=int(os.environ.get("MIN_NEW_RATINGS", "100")),
            threshold_days=int(os.environ.get("MAX_DAYS_WITHOUT_TRAINING", "7"))
        )
        
        logging.info(f"Retrain decision: {retrain_decision}")
        return retrain_decision["should_retrain"]
        
    except Exception as e:
        logging.error(f"Error checking retrain status: {e}")
        return True  # Default to training if check fails

def train_from_mongodb():
    """Train model using MongoDB data (PRODUCTION PATH)"""
    logging.info("🚀 Starting TwoTower model training from MongoDB...")
    
    try:
        # Initialize system
        mr = MovieRecommendationSystem()
        
        # Load data from MongoDB
        df = mr.load_data_from_mongodb(
            days_back=int(os.environ.get("TRAINING_DAYS_BACK", "90")),
            min_ratings_per_user=int(os.environ.get("MIN_RATINGS_USER", "5")),
            min_ratings_per_movie=int(os.environ.get("MIN_RATINGS_MOVIE", "5"))
        )
        
        logging.info(f"Loaded {len(df)} ratings from MongoDB")
        
        # Create datasets with fixed mappings
        train_loader, test_loader, train_df, test_df = mr.create_datasets(
            batch_size=int(os.environ.get("BATCH_SIZE", "64")),
            test_size=float(os.environ.get("TEST_SIZE", "0.2"))
        )
        
        # Create and train model
        mr.create_model()
        device = "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"
        
        mr.train(
            train_loader, test_loader, 
            epochs=int(os.environ.get("EPOCHS", "5")),
            lr=float(os.environ.get("LEARNING_RATE", "0.001")),
            device=device
        )
        
        # Train XGBoost ensemble
        mr.train_xgb(train_loader, test_loader, device=device)
        
        # Save model
        model_path = mr.save_artifact(path=os.environ.get("MODEL_OUT", "artifacts/model.pt"))
        
        logging.info(f"✅ Training completed! Model saved to {model_path}")
        
        # Trigger model reload in API
        try:
            import redis
            redis_client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
            redis_client.publish("model_updated", "reload")
            logging.info("📡 Sent model update signal to workers")
        except Exception as e:
            logging.warning(f"Could not send model update signal: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        return False

def train_from_csv_fallback():
    """Fallback training using CSV files (DEVELOPMENT PATH)"""
    logging.info("⚠️ Falling back to CSV training...")
    
    try:
        mr = MovieRecommendationSystem()
        
        # Load CSV data
        mr.load_data(
            movies_path=os.environ.get("MOVIES_CSV", "data/movies.csv"),
            ratings_path=os.environ.get("RATINGS_CSV", "data/ratings.csv"),
            tags_path=os.environ.get("TAGS_CSV", "data/tags.csv")
        )
        
        # Create datasets
        train_loader, test_loader, train_df, test_df = mr.create_datasets(
            batch_size=int(os.environ.get("BATCH_SIZE", "64"))
        )
        
        # Train model
        mr.create_model()
        device = "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"
        
        mr.train(
            train_loader, test_loader, 
            epochs=int(os.environ.get("EPOCHS", "3")),
            device=device
        )
        
        mr.train_xgb(train_loader, test_loader, device=device)
        mr.save_artifact(path=os.environ.get("MODEL_OUT", "artifacts/model.pt"))
        
        logging.info("✅ CSV fallback training completed")
        return True
        
    except Exception as e:
        logging.error(f"❌ CSV training failed: {e}")
        return False

def main():
    """Main training pipeline with intelligent data source selection"""
    
    logging.info("🎬 TwoTower Movie Recommender Training Pipeline")
    logging.info("=" * 50)
    
    # Check if training is needed
    if not should_train_model():
        logging.info("⏭️ Skipping training - model is up to date")
        return
    
    # Determine data source
    use_mongodb = os.environ.get("USE_MONGODB", "true").lower() == "true"
    
    if use_mongodb:
        # Try MongoDB first (production)
        success = train_from_mongodb()
        
        if not success:
            logging.warning("MongoDB training failed, trying CSV fallback...")
            success = train_from_csv_fallback()
    else:
        # Use CSV directly (development)
        success = train_from_csv_fallback()
    
    if success:
        logging.info("🎉 Training pipeline completed successfully!")
    else:
        logging.error("💥 Training pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main()
