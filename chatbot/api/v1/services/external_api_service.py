import aiohttp
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import hashlib
import os
from urllib.parse import urljoin

from shared.utils.logging_utils import get_module_logger
from pydantic import BaseModel, Field

logger = get_module_logger(__name__)

# ============= PYDANTIC MODELS FOR TYPE SAFETY =============
class ExternalUserProfile(BaseModel):
    """External API user profile response model"""
    id: str
    email: str
    name: str
    phone: str
    created_at: str
    is_active: bool = True

class ExternalChildProfile(BaseModel):
    """External API child profile model"""
    id: str
    name: str
    age: int
    interests: List[str] = []
    created_at: str
    parent_id: str

class ExternalStoryRecord(BaseModel):
    """External API story record model"""
    id: str
    owner_id: str
    title: str
    status: str # "Generating", "completed", "failed"
    child_name: str
    child_age: int
    preferences: Dict[str, Any] = {}
    chapters: List[Dict[str, Any]] = []
    image_urls: List[str] = []
    audio_urls: List[str] = []
    created_at: str
    updated_at: str

class ExternalCommunityPost(BaseModel):
    """External API community post model"""
    post_no: int
    title: str
    writer: str
    content: str
    created_at: str
    views: int = 0

class CreateStoryRequest(BaseModel):
    """Story creation request model"""
    title: str = "AI Generated Fairy Tale"
    status: str = "generating"
    child_name: str
    child_age: int
    preferences: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class UpdateStoryRequest(BaseModel):
    """Story update request model"""
    title: Optional[str] = None
    status: Optional[str] = None
    chapters: Optional[List[Dict[str, Any]]] = None
    image_urls: Optional[List[str]] = None
    audio_urls: Optional[List[str]] = None
    completion_time: Optional[str] = None

class CreatePostRequest(BaseModel):
    """Community post creation request"""
    writer: str
    title: str
    content: str

# ============= CUSTOM EXCEPTIONS =============

class ExternalAPIError(Exception):
    """Base exception for external API errors"""
    def __init__(self, message: str, status_code: int = 0, response_data: Dict = {}):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(self.message)

class AuthenticationError(ExternalAPIError):
    """Authentication related errors"""
    pass

class ResourceNotFoundError(ExternalAPIError):
    """Resource not found errors"""
    pass

class ValidationError(ExternalAPIError):
    """Data validation errors"""
    pass

class ServiceUnavailableError(ExternalAPIError):
    """Service unavailable errors"""
    pass

# ============= MAIN SERVICE CLASS =============
class ExternalAPIService:
    """
    Comprehensive External API Service for JJongal AI

    Handles all communication with the external backend API including:
    - User authentication and profile management
    - Story metadata CRUD operations
    - Community board integration
    - Error handling and resilience
    """

    def __init__(self):
        # Configuration
        self.base_url = os.getenv("EXTERNAL_API_URL", "https://api.jjongal.com")
        self.api_key = os.getenv("EXTERNAL_API_KEY", "")
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.max_retries = 3
        self.retry_delay = 1.0 # seconds

        # Request tracking
        self._request_counter = 0

        logger.info(f"ExternalAPIService initialized with base_url: {self.base_url}")
# ============= CORE HTTP CLIENT METHODS =============  
    async def _make_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Union[Dict, BaseModel]] = None,
            headers: Optional[Dict[str, str]] = None,
            params: Optional[Dict[str, Any]] = None,
            auth_token: Optional[str] = None,
            retries: int = 0
    ) -> Dict[str, Any]:
        """
        Core HTTP request method with comprehensive error handling
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))

        # Build headers
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "JJongal-AI-Service/1.0.0",
            "X-Request-ID": self._generate_request_id()
        }

        if self.api_key:
            request_headers["X-API-Key"] = self.api_key

        if auth_token:
            request_headers["Authorization"] = f"Bearer {auth_token}"
        
        if headers:
            request_headers.update(headers)
        
        # Prepare request data
        json_data = None
        if data:
            if isinstance(data, BaseModel):
                json_data = data.dict(exclude_none=True)
            else:
                json_data = data
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.request(
                        method=method.upper(),
                        url=url,
                        json=json_data,
                        params=params,
                        headers=request_headers
                    ) as response:
                        response_data = await self._handle_response(response)

                        logger.debug(
                            f"API Request successful: {method} {endpoint}"
                            f"-> {response.status} (attempt {attempt + 1})"
                        )

                        return response_data
            
            except aiohttp.ClientError as e:
                logger.warning(f"HTTP client error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise ServiceUnavailableError(
                        f"Failed to connect to external API after {self.max_retries} attempts:{str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))


            except ExternalAPIError:
                # Don't retry on API-level errors
                raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise ExternalAPIError(f"Unexpected error: {str(e)}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response and convert to appropriate exceptions"""
        try:
            if response.content_type.startswith('application/json'):
                response_data = await response.json()
            
            else:
                text_data = await response.text()
                response_data = {"message": text_data}
        except Exception as e:
            response_data = {"message": f"Failed to parse response: {str(e)}"}
            

    