"""
Constants for the CCB-AI project
"""

# AI Models
GPT4O_MINI_MODEL = "gpt-4o-mini"
GPT4O_MODEL = "gpt-4o"
DALLE_MODEL = "dall-e-3"
ELEVENLABS_MODEL = "elevenlabs"

# API Endpoints
OPENAI_API_ENDPOINT = "https://api.openai.com/v1"
ELEVENLABS_API_ENDPOINT = "https://api.elevenlabs.io/v1"

# Audio Settings
SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK_SIZE = 1024

# File Paths
OUTPUT_DIR = "output"
IMAGES_DIR = "images"
AUDIO_DIR = "audio"
PROMPTS_DIR = "data/prompts"
CONVERSATION_DIR = "data/chat"

# Database
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "ccb_ai"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""

# Vector Database
CHROMA_PERSIST_DIR = "data/vector_db"
PINECONE_INDEX = "ccb-ai"


# Cloud Storage
S3_BUCKET = "ccb-ai-storage"
S3_REGION = "ap-northeast-2"

# Noise Filtering
RNNOISE_MODEL_PATH = "data/models/rnnoise.model"
