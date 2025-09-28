# streamlit_app_with_preferences.py - Enhanced with Genre Preferences (FIXED)
import streamlit as st
import requests
import uuid
import os
import pandas as pd
import hashlib
import json
import time
from typing import Dict, List, Optional

# Configuration
API_URL = "http://localhost:8000"

# Available genres for preference selection
AVAILABLE_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Sample movies remain the same
SAMPLE_MOVIES = [
    {"id": "1", "title": "Toy Story (1995)", "genres": ["Animation", "Children", "Comedy"], "year": 1995},
    {"id": "2", "title": "Jumanji (1995)", "genres": ["Adventure", "Children", "Fantasy"], "year": 1995},
    {"id": "3", "title": "Grumpier Old Men (1995)", "genres": ["Comedy", "Romance"], "year": 1995},
    {"id": "4", "title": "Waiting to Exhale (1995)", "genres": ["Comedy", "Drama", "Romance"], "year": 1995},
    {"id": "5", "title": "Father of the Bride Part II (1995)", "genres": ["Comedy"], "year": 1995},
    {"id": "6", "title": "Heat (1995)", "genres": ["Action", "Crime", "Thriller"], "year": 1995},
    {"id": "7", "title": "Sabrina (1995)", "genres": ["Comedy", "Romance"], "year": 1995},
    {"id": "8", "title": "Tom and Huck (1995)", "genres": ["Adventure", "Children"], "year": 1995},
    {"id": "9", "title": "Sudden Death (1995)", "genres": ["Action"], "year": 1995},
    {"id": "10", "title": "GoldenEye (1995)", "genres": ["Action", "Adventure", "Thriller"], "year": 1995}
]

class UserManager:
    """Enhanced user management with preferences and debugging"""
    
    @staticmethod
    def create_user_id(username: str) -> str:
        """Create a consistent user ID from username"""
        return hashlib.md5(username.encode()).hexdigest()[:12]
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[str]:
        """Simple authentication - in production, use proper auth"""
        if len(username) >= 3 and len(password) >= 4:
            return UserManager.create_user_id(f"{username}:{password}")
        return None
    
    @staticmethod
    def check_api_health() -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"API Health Check Failed: {e}")
            return False
    
    @staticmethod
    def check_user_preferences(user_id: str) -> bool:
        """Check if user has set genre preferences"""
        try:
            if not UserManager.check_api_health():
                st.error("API server is not running. Please start the FastAPI server.")
                return False
                
            response = requests.get(f"{API_URL}/user/{user_id}/preferences", timeout=15)
            if response.status_code == 200:
                prefs = response.json()
                return prefs.get("has_preferences", False)
            else:
                st.warning(f"Could not check preferences: {response.status_code}")
                return False
        except Exception as e:
            st.error(f"Error checking preferences: {e}")
            return False
    
    @staticmethod
    def save_user_preferences(user_id: str, preferred_genres: List[str], 
                            disliked_genres: List[str] = None) -> bool:
        """Save user genre preferences with proper error handling"""
        try:
            # First check if API is reachable
            if not UserManager.check_api_health():
                st.error("❌ API server is not running. Please start the FastAPI server on port 8000.")
                return False
            
            payload = {
                "user_id": user_id,
                "preferred_genres": preferred_genres,
                "disliked_genres": disliked_genres or [],
                "preference_strength": 1.0
            }
            
            # Show debug info in expander
            with st.expander("Debug Information", expanded=False):
                st.write(f"API URL: {API_URL}")
                st.write(f"User ID: {user_id}")
                st.write(f"Payload: {payload}")
            
            # FIXED: Use the correct endpoint path
            response = requests.post(f"{API_URL}/user/preferences", json=payload, timeout=15)
            
            st.write(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Preferences saved successfully!")
                st.write(f"Server response: {result}")
                return True
            else:
                st.error(f"❌ Failed to save preferences. Status code: {response.status_code}")
                try:
                    error_detail = response.json()
                    st.error(f"Error details: {error_detail}")
                except:
                    st.error(f"Raw response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The server might be slow or overloaded.")
            return False
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API server. Make sure it's running on http://localhost:8000")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return False

class MovieService:
    """Handle movie data and poster fetching"""
    
    def __init__(self):
        self.movies_cache = {movie["id"]: movie for movie in SAMPLE_MOVIES}
    
    def get_movies_by_genres(self, preferred_genres: List[str], limit: int = 10) -> List[Dict]:
        """Get movies matching preferred genres"""
        matching_movies = []
        for movie in SAMPLE_MOVIES:
            # Check if movie has any of the preferred genres
            if any(genre in movie["genres"] for genre in preferred_genres):
                matching_movies.append(movie)
        
        # Sort by number of matching genres (more matches = better)
        matching_movies.sort(
            key=lambda m: sum(1 for g in m["genres"] if g in preferred_genres),
            reverse=True
        )
        
        return matching_movies[:limit]
    
    def get_movie_by_id(self, movie_id: str) -> Optional[Dict]:
        return self.movies_cache.get(movie_id)
    
    def search_movies(self, query: str) -> List[Dict]:
        query = query.lower()
        return [movie for movie in SAMPLE_MOVIES 
                if query in movie["title"].lower()]
    
    def get_placeholder_poster(self, movie_title: str) -> str:
        title_encoded = movie_title.replace(" ", "+")
        return f"https://via.placeholder.com/300x450/4A90E2/FFFFFF?text={title_encoded}"

# Initialize services
movie_service = MovieService()
user_manager = UserManager()

# Streamlit App
st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

st.title("🎬 MovieLens Recommender System")
st.markdown("*Discover movies you'll love based on your ratings and preferences*")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
    st.session_state["username"] = None
    st.session_state["preferences_set"] = False

# User Authentication Section
st.sidebar.header("👤 User Account")

if not st.session_state["user_id"]:
    st.sidebar.markdown("**Create Account or Login**")
    
    with st.sidebar.form("user_auth"):
        username = st.text_input("Username", placeholder="Enter unique username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("Login"):
                user_id = user_manager.authenticate_user(username, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    # Check if user has preferences
                    st.session_state["preferences_set"] = user_manager.check_user_preferences(user_id)
                    st.sidebar.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
        
        with col2:
            if st.form_submit_button("Register"):
                user_id = user_manager.authenticate_user(username, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    st.session_state["preferences_set"] = False  # New user needs to set preferences
                    st.sidebar.success(f"Account created for {username}!")
                    st.rerun()
                else:
                    st.sidebar.error("Username/password too short")

else:
    # User is logged in
    st.sidebar.success(f"👋 Hello, **{st.session_state['username']}**!")
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.session_state["preferences_set"] = False
        st.rerun()

# API Status Check
with st.sidebar:
    st.markdown("---")
    if st.button("🔍 Check API Status"):
        if user_manager.check_api_health():
            st.success("✅ API Server is running")
        else:
            st.error("❌ API Server is offline")

# Genre Preferences Onboarding (for new users or users without preferences)
if st.session_state["user_id"] and not st.session_state["preferences_set"]:
    st.markdown("## 🎭 Welcome! Let's Personalize Your Experience")
    st.markdown("""
    To give you better movie recommendations, please tell us about your genre preferences.
    This will help us suggest movies you're more likely to enjoy!
    """)
    
    with st.form("genre_preferences"):
        st.markdown("### Select Your Favorite Genres (Choose 3-5):")
        
        # Create columns for better layout
        cols = st.columns(3)
        selected_genres = []
        
        for i, genre in enumerate(AVAILABLE_GENRES):
            with cols[i % 3]:
                if st.checkbox(genre, key=f"pref_{genre}"):
                    selected_genres.append(genre)
        
        st.markdown("### Genres You Don't Like (Optional):")
        cols_dislike = st.columns(3)
        disliked_genres = []
        
        for i, genre in enumerate(AVAILABLE_GENRES):
            with cols_dislike[i % 3]:
                if st.checkbox(f"Dislike {genre}", key=f"dislike_{genre}"):
                    disliked_genres.append(genre)
        
        # Submit preferences
        if st.form_submit_button("Save My Preferences", type="primary"):
            if len(selected_genres) < 2:
                st.error("Please select at least 2 favorite genres.")
            elif len(selected_genres) > 8:
                st.error("Please select no more than 8 genres to keep recommendations focused.")
            else:
                # Save preferences to backend
                with st.spinner("Saving preferences..."):
                    success = user_manager.save_user_preferences(
                        st.session_state["user_id"], 
                        selected_genres, 
                        disliked_genres
                    )
                
                if success:
                    st.session_state["preferences_set"] = True
                    st.success("Preferences saved! Getting your personalized recommendations...")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
    
    # Show sample recommendations based on selected genres
    if len([genre for genre in AVAILABLE_GENRES if st.session_state.get(f"pref_{genre}", False)]) > 0:
        selected_preview_genres = [genre for genre in AVAILABLE_GENRES if st.session_state.get(f"pref_{genre}", False)]
        st.markdown("### Preview: Movies You Might Like")
        sample_movies = movie_service.get_movies_by_genres(selected_preview_genres, 6)
        
        if sample_movies:
            cols = st.columns(3)
            for i, movie in enumerate(sample_movies):
                with cols[i % 3]:
                    st.image(movie_service.get_placeholder_poster(movie['title']), width=150)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"*{', '.join(movie['genres'][:2])}*")

# Main App Content (only show if preferences are set)
elif st.session_state["user_id"] and st.session_state["preferences_set"]:
    
    # Movie Rating Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎭 Rate a Movie")
        
        search_query = st.text_input("🔍 Search for movies", placeholder="Type movie title...")
        
        if search_query:
            search_results = movie_service.search_movies(search_query)
            if search_results:
                movie_options = [f"{movie['title']}" for movie in search_results]
                selected_movie_title = st.selectbox("Select Movie:", movie_options)
                selected_movie = next(movie for movie in search_results if movie['title'] == selected_movie_title)
            else:
                st.warning("No movies found. Try a different search term.")
                selected_movie = None
        else:
            st.markdown("**Popular Movies:**")
            movie_options = [f"{movie['title']}" for movie in SAMPLE_MOVIES[:10]]
            selected_movie_title = st.selectbox("Select Movie:", movie_options)
            selected_movie = next(movie for movie in SAMPLE_MOVIES if movie['title'] == selected_movie_title)
    
    with col2:
        if 'selected_movie' in locals() and selected_movie:
            st.markdown("### Movie Details")
            st.image(movie_service.get_placeholder_poster(selected_movie['title']), width=200)
            st.markdown(f"**Title:** {selected_movie['title']}")
            st.markdown(f"**Year:** {selected_movie['year']}")
            st.markdown(f"**Genres:** {', '.join(selected_movie['genres'])}")
    
    # Rating Section
    if 'selected_movie' in locals() and selected_movie:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            rating = st.slider("⭐ Rate this movie", 0.5, 5.0, 3.0, 0.5)
            comment = st.text_area("💬 Your thoughts (optional)", placeholder="What did you think about this movie?")
        
        with col2:
            stars_display = "⭐" * int(rating) + "☆" * (5 - int(rating))
            st.markdown(f"### Your Rating: {rating}/5.0")
            st.markdown(f"### {stars_display}")
        
        if st.button("🎯 Submit Rating", type="primary"):
            try:
                payload = {
                    "user_id": st.session_state["user_id"],
                    "movie_id": selected_movie["id"],
                    "rating": float(rating),
                    "comment": comment,
                    "tag": comment,  # Use comment as tag
                    "genre": ", ".join(selected_movie["genres"])
                }
                
                response = requests.post(f"{API_URL}/submit_rating", json=payload, timeout=15)
                if response.status_code == 200:
                    st.success(f"✅ Rating submitted for {selected_movie['title']}!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed to submit rating: {response.status_code}")
                    st.error(f"Response: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {e}")

    # Recommendations Section
    st.markdown("---")
    st.header("🎯 Your Personal Recommendations")
    
    try:
        response = requests.get(f"{API_URL}/recommendations/{st.session_state['user_id']}", timeout=15)
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get("recommendations", [])
            
            if recommendations:
                st.markdown(f"*Based on your preferences and ratings, here are personalized recommendations:*")
                
                # Display recommendations
                num_cols = 3
                cols = st.columns(num_cols)
                
                for i, rec in enumerate(recommendations):
                    with cols[i % num_cols]:
                        movie_details = movie_service.get_movie_by_id(rec.get("movie_id", ""))
                        
                        if movie_details:
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                                <h4 style="margin: 0; color: #333;">{movie_details['title']}</h4>
                                <p style="margin: 5px 0; color: #666;"><strong>Genres:</strong> {', '.join(movie_details['genres'][:2])}</p>
                                <p style="margin: 5px 0; color: #4A90E2;"><strong>Match Score:</strong> {rec.get('predicted_rating', 0):.1f}/5.0</p>
                                <p style="margin: 5px 0; color: #28a745;"><strong>Rank:</strong> #{rec.get('rank', i+1)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.image(movie_service.get_placeholder_poster(movie_details['title']), width=150)
                        else:
                            st.markdown(f"**Movie ID:** {rec.get('movie_id', 'Unknown')}")
                            st.markdown(f"**Score:** {rec.get('predicted_rating', 0):.1f}")
            else:
                st.info("🎬 Rate a few more movies to get better recommendations!")
        else:
            st.warning(f"⚠️ Could not load recommendations. Status: {response.status_code}")
            st.error(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to recommendation service: {e}")

    # Preferences Management
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Preferences")
        if st.button("🔧 Update Genre Preferences"):
            st.session_state["preferences_set"] = False
            st.rerun()

elif st.session_state["user_id"]:
    # User logged in but preferences not set (shouldn't happen with current flow)
    st.info("Setting up your preferences...")
    st.session_state["preferences_set"] = False
    st.rerun()

else:
    # User not logged in - welcome screen
    st.markdown("""
    ## 🎬 Welcome to MovieLens Recommender!
    
    This intelligent movie recommendation system learns from your ratings and genre preferences 
    to suggest films you'll love.
    
    ### How it works:
    1. **Create an account** using the sidebar
    2. **Set your genre preferences** to get personalized recommendations from day one
    3. **Rate movies** you've seen to improve recommendations
    4. **Discover new favorites** with our smart algorithms
    
    ### Features:
    - 🎭 **Genre-Based Cold Start** - Immediate recommendations for new users
    - 🤖 **AI-Powered Learning** - Recommendations improve with every rating
    - 🎯 **Real-time Updates** - Fresh suggestions as your taste evolves
    - 📊 **Personal Insights** - Track your preferences and discoveries
    
    **👈 Get started by creating an account in the sidebar!**
    """)
    
    # Show featured movies
    st.markdown("### Featured Movies")
    cols = st.columns(5)
    for i, movie in enumerate(SAMPLE_MOVIES[:5]):
        with cols[i]:
            st.image(movie_service.get_placeholder_poster(movie['title']), width=120, caption=movie['title'])

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and advanced ML algorithms*")