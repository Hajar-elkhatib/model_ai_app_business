"""
chatbot_db.py
──────────────
MongoDB connector for the VentureLens Business Chatbot.
Fetches user and business data to personalize the chat experience.
"""

import logging
from typing import Any
from pymongo import MongoClient
from chatbot_config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)

# Initialize client lazily
_client = None
_db = None

def get_db():
    """Get or initialize the MongoDB database connection."""
    global _client, _db
    if _db is not None:
        return _db
        
    if not MONGODB_URI:
        logger.warning("MONGODB_URI is not set. Database features will be disabled.")
        return None
        
    try:
        # Use a reasonable timeout so we don't hang if DB is unreachable
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _db = _client[MONGODB_DB_NAME]
        # Force a connection check
        _client.server_info()
        logger.info(f"Connected to MongoDB database: {MONGODB_DB_NAME}")
        return _db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        _db = None
        _client = None
        return None

def get_business_data(user_id: str) -> dict[str, Any] | None:
    """
    Fetch business/project data for a given user.
    
    NOTE: You may need to change collection names and query fields
    to match your Spring Boot / MongoDB schema.
    """
    db = get_db()
    if db is None:
        return None
        
    try:
        # Assuming a collection named 'projects' and that it has a field 'userId'
        # Adjust as needed for your database schema!
        project = db["projects"].find_one({"userId": user_id})
        
        if project:
            # Remove MongoDB's default _id if it's an ObjectId (not JSON serializable)
            if "_id" in project:
                project["_id"] = str(project["_id"])
            return project
            
        return None
    except Exception as e:
        logger.error(f"Error fetching business data: {e}")
        return None

def get_user_data(user_id: str) -> dict[str, Any] | None:
    """Fetch user profile data."""
    db = get_db()
    if db is None:
        return None
        
    try:
        # Assuming a collection named 'users'
        user = db["users"].find_one({"_id": user_id})
        if user:
            if "_id" in user:
                user["_id"] = str(user["_id"])
            return user
        return None
    except Exception as e:
        logger.error(f"Error fetching user data: {e}")
        return None
