# config/api_config.py


import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIConfig:
    """All API configuration in one place. Add keys when you get them."""
    
    # Gemini
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = 'gemini-1.5-flash'  # or pro
    GEMINI_MAX_TOKENS = 4096
    GEMINI_TEMPERATURE = 0.1  # Low for consistent JSON
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = 'gpt-4-vision-preview'
    OPENAI_MAX_TOKENS = 4096
    OPENAI_TEMPERATURE = 0.1
    
    # Common settings
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    # Cost tracking (update with actual prices)
    COST_PER_1K_TOKENS = {
        'gemini-flash': 0.0005,  # Approx - update when you get API
        'gpt4-vision': 0.01       # Approx - update when you get API
    }
    
    @classmethod
    def is_gemini_ready(cls):
        return bool(cls.GEMINI_API_KEY)
    
    @classmethod
    def is_openai_ready(cls):
        return bool(cls.OPENAI_API_KEY)