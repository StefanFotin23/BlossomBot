import os
import redis.asyncio as redis
from typing import Optional

# Global variable to hold the Redis client instance
redis_client: Optional[redis.Redis] = None

def get_redis_client() -> redis.Redis:
    """
    Returns a Redis client instance.
    Initializes the client if it's not already created.
    """
    global redis_client
    if redis_client is None:
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_client = redis.from_url(f"redis://{redis_host}")
    return redis_client
