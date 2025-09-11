import os
import jwt
import asyncio

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Type, TypeVar
from functools import lru_cache, wraps
from fastapi import WebSocket, WebSocketException, Query, Depends, status, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager

from src.shared.utils.logging import get_module_logger
from src.shared.configs.app import get_env_vars, get_app_settings
from src.shared.utils.websocket import ConnectionManager

logger = get_module_logger(__name__)

T = TypeVar("T")

# ======= Authentication Dependencies ======


class AuthenticationError(WebSocketException):
    def __init__(self, reason: str = "Authentication Failed"):
        super().__init__(code=status.WS_1008_POLICY_VIOLATION, reason=reason)


class HTTPAuthenticationError(HTTPException):
    def __init__(self, detail: str = "Authentication Failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class JWTManager:
    def __init__(self):
        self.env_vars = get_env_vars()
        self.secret_key = self.env_vars.get("jwt_secret_key", "jjongal_default_secret")
        self.algorithm = "HS256"
        self.default_expiry_hours = 24

    def generate_token(self, payload: Dict[str, Any], expiry_hours: int = None) -> str:
        expiry_hours = expiry_hours or self.default_expiry_hours
        expiry = datetime.utcnow() + timedelta(hours=expiry_hours)

        token_data = {"exp": expiry, "iat": datetime.utcnow(), **payload}

        return jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

    def create_development_token(self, user_id: str = "dev_user") -> str:
        payload = {
            "user_id": user_id,
            "user_name": "dev_user",
            "role": "developer",
            "dev": True,
        }
        return self.generate_token(payload=payload)


# Singleton JWT manager
@lru_cache()
def get_jwt_manager() -> JWTManager:
    return JWTManager()


async def verify_jwt_token_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token"),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> Dict[str, Any]:
    if not token:
        logger.warning("Websocket connection attempt : no token")
        raise AuthenticationError("JWT token required")

    try:
        user_info = jwt_manager.verify_token(token)
        logger.info(f"WebSocket auth success: {user_info.get('user_id', 'unknown')}")
        return user_info
    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket auth failed: token expired")
        raise AuthenticationError("Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"WebSocket auth failed: invalid token - {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"WebSocket auth error: {e}")
        raise AuthenticationError("Authentication error occured")


# ==== Service Factory Pattern ====
class ServiceFactory:
    _instances: Dict[str, Any] = {}
    _initializers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, service_name: str, initializer: Callable) -> None:
        cls._initializers[service_name] = initializer

    @classmethod
    async def get_instance(cls, service_name: str) -> Any:
        # Get service instance (singleton)
        if service_name not in cls._instances:
            if service_name not in cls._initializers:
                raise ValueError(f"Unknown serivce: {service_name}")

            initializer = cls._initializers[service_name]
            if asyncio.iscoroutinefunction(initializer):
                cls._instances[service_name] = await initializer()
            else:
                cls._instances[service_name] = initializer()

        return cls._instances[service_name]


# Service initializers
def _init_connection_manager() -> ConnectionManager:
    app_settings = get_app_settings()
    timeout = app_settings.get("ws_connection_timeout", 1800)
    return ConnectionManager(connection_timeout=timeout)


def _init_chatbot_a():
    from src.core.chatbots.chat_bot_a.chat_bot_a import ChatBotA

    return ChatBotA(model_name="gpt-4o-mini", temperature=0.8, enable_monitoring=True)


def _init_voice_cloning_processor():
    from src.core.voice.processors.voice_cloning_processor import VoiceCloningProcessor

    return VoiceCloningProcessor()


def _init_jjong_ari_collaborator():
    from src.core.chatbots.collaboration.jjong_ari_collaborator import (
        ModernJjongAriCollaborator,
    )

    return ModernJjongAriCollaborator()


def _init_audio_processor():
    from src.core.voice.processors.audio_processor import AudioProcessor

    return AudioProcessor()


# Register services
ServiceFactory.register("connection_manager", _init_connection_manager)
ServiceFactory.register("chatbot_a", _init_chatbot_a)
ServiceFactory.register("voice_processor", _init_voice_cloning_processor)
ServiceFactory.register("collaborator", _init_jjong_ari_collaborator)
ServiceFactory.register("audio_processor", _init_audio_processor)

# ==== Dependency Injection Functions ====


async def get_connection_manager() -> ConnectionManager:
    return await ServiceFactory.get_instance("connection_manager")


async def get_chatbot_a():
    return await ServiceFactory.get_instance("chatbot_a")


async def get_voice_cloning_processor():
    return await ServiceFactory.get_instance("voice_processor")


async def get_jjong_ari_collaborate():
    return await ServiceFactory.get_instance("collaborator")


async def get_audio_processor():
    return await ServiceFactory.get_instance("audio_processor")


# ==== Context Managers ====


@asynccontextmanager
async def websocket_session_context(
    client_id: str, websocket: WebSocket, connection_manager: ConnectionManager
):
    # Websocket session context manager - Auto-manage connection lifecycle
    try:
        await connection_manager.connect(websocket, client_id)
        logger.info(f"WebSocket session started: {client_id}")
        yield connection_manager
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
        raise
    finally:
        connection_manager.disconnect(client_id=client_id)
        logger.info(f"WebSocket session ended: {client_id}")


# ==== Decorators ====
def Websocket_error_handler(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"WebSocket handler error ({func.__name__}): {e}")
            raise

    return wrapper


# ====== REST API Dependencies ======

# HTTPBearer security scheme for automatic token extraction
security = HTTPBearer()

async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> Dict[str, Any]:
    """REST API JWT token verification"""
    token = credentials.credentials
    if not token:
        logger.warning("REST API request: no token provided")
        raise HTTPAuthenticationError("JWT token required")

    try:
        user_info = jwt_manager.verify_token(token)
        logger.info(f"REST API auth success: {user_info.get('user_id', 'unknown')}")
        return user_info
    except jwt.ExpiredSignatureError:
        logger.warning("REST API auth failed: token expired")
        raise HTTPAuthenticationError("Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"REST API auth failed: invalid token - {e}")
        raise HTTPAuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"REST API auth error: {e}")
        raise HTTPAuthenticationError("Authentication error occurred")


async def get_authenticated_user(
    jwt_manager: JWTManager = Depends(get_jwt_manager)
) -> Dict[str, Any]:
    """Get authenticated user from JWT token (for development)"""
    # For development/demo purposes, return a default user
    # In production, this would extract token from headers
    try:
        dev_token = jwt_manager.create_development_token()
        user_info = jwt_manager.verify_token(dev_token)
        return user_info
    except Exception as e:
        logger.error(f"Failed to create development user: {e}")
        raise HTTPAuthenticationError("Authentication failed")

async def get_vector_db():
    """Get LFU Hybrid vector database instance - 2025 Advanced Architecture"""
    try:
        from src.data.vector_db.core import LFUHybridVectorDB, SearchWeights
        import os
        
        # Get vector DB path from environment or use default
        vector_db_path = os.getenv("VECTOR_DB_PATH", "data/chroma_db")
        
        logger.info("🚀 Initializing LFU + Hybrid ChromaDB system...")
        
        # Initialize LFU Hybrid VectorDB with optimized settings
        lfu_vector_db = LFUHybridVectorDB(
            persist_directory=vector_db_path,
            collection_name="fairy_tales",
            use_openai=False,  # Use KURE-v1 Korean embeddings
            memory_cache_size=1000,  # Large cache for production
            disk_cache_size=5000,    # Even larger disk cache
            disk_cache_ttl=86400,    # 24 hour TTL
            enable_bm25=True,        # Enable hybrid search
            search_weights=SearchWeights(
                vector_weight=0.5,    # Vector similarity
                bm25_weight=0.3,      # Keyword matching
                metadata_weight=0.2   # Category/metadata filtering
            )
        )
        
        logger.info("✅ LFU Hybrid VectorDB initialized successfully")
        
        # Get collection statistics
        try:
            collection_count = lfu_vector_db.vectorstore._collection.count() if hasattr(lfu_vector_db.vectorstore, '_collection') else 0
        except:
            collection_count = 0
        
        # Get cache statistics
        cache_stats = lfu_vector_db.get_cache_stats()
        
        logger.info(f"📊 Vector DB Status: {collection_count} documents, Cache: {cache_stats['cache_stats']['cache_size']} entries")
        
        return {
            "status": "connected",
            "type": "lfu_hybrid_chromadb",
            "vector_db": lfu_vector_db,
            "vectorstore": lfu_vector_db.vectorstore,
            "path": vector_db_path,
            "collection_count": collection_count,
            "cache_stats": cache_stats,
            "hybrid_features": {
                "lfu_cache": True,
                "bm25_search": lfu_vector_db.enable_bm25,
                "korean_embeddings": "KURE-v1",
                "multi_layer_caching": True
            }
        }
        
    except ImportError as e:
        logger.error(f"Vector DB dependencies not available: {e}")
        return {
            "status": "error",
            "type": "missing_dependencies",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Vector DB connection failed: {e}")
        return {
            "status": "error", 
            "type": "connection_failed",
            "error": str(e)
        }

# Note: Voice cloning processor is now handled via ServiceFactory pattern above

# ====== Development / Testing Support ======


async def verify_development_token(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Development token"),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> Dict[str, Any]:
    # Development simple authentication
    env_vars = get_env_vars()

    if os.getenv("ENVIRONMENT") != "development":
        raise AuthenticationError("Available only in development mode")

    if not token:
        raise AuthenticationError("Development token required")

    # Support fixed dev token or JWT token
    if token == env_vars.get("ws_auth_token", "dev-token"):
        return {
            "user_id": "dev_user",
            "username": "dev_user",
            "role": "developer",
            "dev": True,
        }

    try:
        return jwt_manager.verify_token(token)
    except Exception:
        raise AuthenticationError("Invalid development token")
