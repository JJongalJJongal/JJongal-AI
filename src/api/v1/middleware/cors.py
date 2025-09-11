"""
CORS Middleware Configuration for JJongal-AI API

Cross-Origin Resource Sharing setup for enabling frontend access
to the JJongal-AI storytelling API endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)


def setup_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None
):
    """
    Setup CORS middleware for FastAPI application
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins, defaults to development origins
        allow_credentials: Whether to allow credentials in CORS requests
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
    """
    
    # Default allowed origins for development
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # React development server
            "http://localhost:8080",  # Vue.js development server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://localhost:8000",  # FastAPI development server
            "http://127.0.0.1:8000",
        ]
    
    # Default allowed methods
    if allowed_methods is None:
        allowed_methods = [
            "GET",
            "POST", 
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD"
        ]
    
    # Default allowed headers
    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "X-External-Token",
            "Cache-Control",
            "Pragma"
        ]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        expose_headers=["X-Total-Count", "X-Page-Count"]
    )
    
    logger.info(f"CORS middleware configured with {len(allowed_origins)} allowed origins")
    logger.debug(f"Allowed origins: {allowed_origins}")


def setup_cors_for_development(app: FastAPI):
    """
    Setup permissive CORS for development environment
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    logger.warning("CORS configured for development - allowing all origins")


def setup_cors_for_production(
    app: FastAPI,
    production_origins: List[str]
):
    """
    Setup restrictive CORS for production environment
    
    Args:
        app: FastAPI application instance
        production_origins: List of allowed production origins
    """
    setup_cors(
        app=app,
        allowed_origins=production_origins,
        allow_credentials=True,
        allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allowed_headers=[
            "Accept",
            "Content-Type", 
            "Authorization",
            "X-Requested-With"
        ]
    )
    
    logger.info(f"CORS configured for production with {len(production_origins)} allowed origins")